import numpy as np
import pandas as pd
import os
import torch
from .dataloader import Dataset_handler
from .predict import load_model


def test(model, dataloader):
    """
    test one model
    """
    model.network.eval()
    gt = []
    for input_batch, target_batch in dataloader:
        gt.append(np.array(target_batch))
        _ = model.evaluate(input_batch, target_batch)
    gts = np.vstack(gt)
    scores = model.results_val["scores"]
    ids = [os.path.splitext(os.path.basename(x))[0] for x in dataloader.dataset.files]
    outputs_dict = {
        "gt": gts,
        "scores": scores,
        "ids": ids,
        "label_encoder": model.label_encoder,
    }
    return outputs_dict


def fill_table(table, proba_preds, preds, ids):
    """
    returns the "data_table" with the additional columsn scores and preds.
    scores, preds, ids are lists, indices correspond to the same image.
    """
    pi_scores = []
    pi_preds = []

    def is_in_set(x):
        if x["ID"] in ids:
            return True
        else:
            return False

    table["take"] = table.apply(is_in_set, axis=1)
    table = table[table["take"]]
    for i in table["ID"].values:
        index = ids.index(i)
        pi_scores.append(proba_preds[index])
        pi_preds.append(preds[index])
    table["proba_preds"] = pi_scores
    table["prediction"] = pi_preds
    return table


def main(model_path=None, rm_duplicates=True, verbose=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)
    model.network.print_summary(verbose=verbose)
    args = model.args
    table = pd.read_csv(args.target_path)
    if rm_duplicates:  # Allows the use of upsampling.
        table = table.drop_duplicates()
    args.train = False
    data = Dataset_handler(args)
    dataloader = data.get_loader(training=False)
    results = test(model, dataloader)
    return results
