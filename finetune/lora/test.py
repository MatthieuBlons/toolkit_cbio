import numpy as np
import os
from finetune.lora.dataloader import Dataset_handler
from finetune.lora.predict import load_model_from_path
import torch
from tqdm import tqdm
from finetune.lora.utils import print_dict


def test(model, dataloader):
    """
    test model
    """
    model.network.eval()
    gt = []

    progress = tqdm(
        desc=f"test...",
        total=len(dataloader),
        unit="batch",
        initial=0,
        leave=False,
    )

    for input_batch, target_batch in dataloader:
        gt.append(np.array(target_batch.squeeze()))
        loss = model.evaluate(input_batch, target_batch)
        progress.set_postfix_str(
            f"{model.args.criterion} loss = {loss:.4}", refresh=True
        )
        progress.update()

    gts = np.vstack(gt)
    pred = np.vstack(model.results_val["pred"])

    img_paths = [os.path.basename(x) for x in dataloader.dataset.input_files]
    features = dataloader.dataset.target_lables
    outputs_dict = {
        "gt": gts,
        "pred": pred,
        "img_path": img_paths,
        "features": features,
    }
    return outputs_dict


def main(model_path: str | None = None, input_path: str | None = None, verbose=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device used = {device}")
    model = load_model_from_path(model_path, device)
    if model.network.lora and (verbose > 0):
        model.network.print_lora_summary()
        model.network.print_summary(verbose=verbose)
    args = model.args
    args.train = False
    if input_path is not None:
        args.input_path = input_path
    print_dict(args.__dict__, name="model args")
    data = Dataset_handler(
        args, img_only=False, preprocess=model.network.transform, predict=False
    )
    dataloader = data.get_loader(training=False)
    results = test(model, dataloader)
    return results, args
