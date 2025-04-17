from .arguments import get_arguments
from .models import DeepMIL
import numpy as np
import torch
from tqdm import tqdm


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
    # check if dataloader use all wsi even when lent(train_data)%batch_size!=0
    epobatch = 1 / len(dataloader)
    for input_batch, target_batch in dataloader:
        # Feed the network with a batch and optimize the parameter
        model.counter["batch"] += 1
        model.counter["epoch"] += epobatch
        loss = model.optimize_parameters(input_batch, target_batch)
        mean_loss.append(loss)
    model.mean_train_loss = np.mean(mean_loss)


def val(model, dataloader):
    model.network.eval()
    mean_loss = []
    for input_batch, target_batch in dataloader:
        target_batch = target_batch.to(model.device)
        loss = model.evaluate(input_batch, target_batch)
        mean_loss.append(loss)
    model.mean_val_loss = np.mean(mean_loss)
    to_write = model.flush_val_metrics()
    writes_metrics(model.writer, to_write, model.counter["epoch"])
    state = model.make_state()
    model.update_learning_rate(model.mean_val_loss)
    model.early_stopping(model.args.sgn_metric * to_write[model.args.ref_metric], state)


def main(known_args=None, verbose=False):
    args = get_arguments(known_args=known_args, train=True)
    model = DeepMIL(args=args, with_data=True)
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
            f"lr={lrs[0]:.3}, train_loss={model.mean_train_loss:.4}, val_loss={model.mean_val_loss:.4}, BEST {model.ref_metric}={best_ref_metric:.2}, ON EPOCH={int(best_epoch)}",
            refresh=True,
        )
        progress.update()
    model.writer.close()
