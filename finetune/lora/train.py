from finetune.lora.arguments import get_arguments
from finetune.lora.models import HESingIF
import numpy as np
import torch
import wandb
import signal
from tqdm import tqdm
import os
import sys


def handle_exit(signum, frame):
    print(f"[INFO] Caught signal {signum}, finishing wandb run...")
    wandb.finish()
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)


def writes_metrics(writer, to_write, epoch):
    """writes_metrics.
    Writes the validation metrics (and the train loss) in a Tensorboard Writer.

    :param writer: Tensorboard Writer
    :param to_write: dict, scalars to write.
    :param epoch: time step.
    """
    for key in to_write:
        if type(to_write[key]) == dict:
            writer.add_scalars(key, to_write[key], epoch)
        else:
            writer.add_scalar(key, to_write[key], epoch)


def train(model, dataloader):

    model.network.train()
    mean_loss = []
    epobatch = 1 / len(dataloader)

    progress = tqdm(
        desc=f"training...",
        total=len(dataloader),
        unit="step",
        initial=0,
        leave=False,
    )
    for input_batch, target_batch in dataloader:
        # Feed the network with a batch and optimize the parameter
        model.counter["epoch"] += epobatch
        model.counter["step"] += 1
        loss = model.optimize_parameters(input_batch, target_batch)
        mean_loss.append(loss)
        progress.set_postfix_str(
            f"{model.args.criterion} loss = {loss:.4}", refresh=True
        )
        progress.update()
        if model.counter["step"] % 100 == 0:
            wandb.log(
                {
                    "train_loss": loss,
                    "lr": model.schedulers[0]._last_lr[0],
                },
                step=model.counter["step"],
            )
    
    progress.close()
    model.mean_train_loss = np.mean(mean_loss)


def val(model, dataloader):
    model.network.eval()
    mean_loss = []

    progress = tqdm(
        desc=f"validation...",
        total=len(dataloader),
        unit="step",
        initial=0,
        leave=False,
    )

    for input_batch, target_batch in dataloader:
        target_batch = target_batch.to(model.device)
        loss = model.evaluate(input_batch, target_batch)
        mean_loss.append(loss)
        progress.set_postfix_str(
            f"{model.args.criterion} loss = {loss:.4}", refresh=True
        )
        progress.update()

    progress.close()
    model.mean_val_loss = np.mean(mean_loss)
    to_write = model.flush_val_metrics()
    wandb.log(
        {
            **to_write,
            "epoch": model.counter["epoch"],
        },
        step=model.counter["step"],
    )
    writes_metrics(model.writer, to_write, model.counter["epoch"])
    state = model.make_state()

    if model.args.lr_scheduler == "linear":
        model.update_learning_rate(model.mean_val_loss)
    elif model.args.lr_scheduler == "cos":
        model.update_learning_rate(None)
    model.early_stopping(model.args.sgn_metric * to_write[model.args.ref_metric], state)


def main(project, job, known_args=None, verbose=False):
    args = get_arguments(known_args=known_args, train=True)

    if int(os.environ.get("RANK", 0)) == 0:
        wandb.init(
            project=project,
            group=job,
            name=f"test_{args.test_fold}_rep_{args.repeat}",
            config=vars(args),
            reinit="return_previous",
        )
    else:
        wandb.init(mode="disabled")

    model = HESingIF(args=args, with_data=True)
    model.get_summary_writer()
    progress = tqdm(
        desc=f"train rep={args.repeat+1}/{args.reps}, fold={args.test_fold+1}/{args.k_folds}",
        total=args.epochs,
        unit="epoch",
        initial=0,
        leave=False,
        disable=not verbose,
    )
    while model.counter["epoch"] < args.epochs:
        train(model=model, dataloader=model.train_loader)
        if args.use_val:
            val(model=model, dataloader=model.val_loader)
        if model.early_stopping.early_stop:
            break
        if not args.use_val:
            torch.save(model.make_state(), "model_best.pt.tar")
        if model.early_stopping.is_best:
            best_epoch = model.counter["epoch"]
            best_ref_metric = model.best_ref_metric
        lrs = [scheduler._last_lr[0] for scheduler in model.schedulers]
        progress.set_postfix_str(
            f"lr={lrs[0]:.3}, train_loss={model.mean_train_loss:.4}, val_loss={model.mean_val_loss:.4}, BEST {model.ref_metric}={best_ref_metric:.4}, ON EPOCH={int(best_epoch)}",
            refresh=True,
        )
        progress.update()
    stop_epoch = int(model.counter["epoch"])
    model.writer.close()

    wandb.finish()
    return stop_epoch
