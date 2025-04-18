"""
Test a given model.
"""

import numpy as np
import pandas as pd
import os
import torch
from .dataloader import Dataset_handler
from .predict import load_model
from collections.abc import MutableMapping


def convert_flatten(d, parent_key="", sep="_"):
    """
    Flattens a nested dict.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(convert_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def test(model, dataloader, table):
    """
    test one model
    """
    model.network.eval()
    gt = []
    for input_batch, target_batch in dataloader:
        gt.append(np.array(target_batch))
        _ = model.evaluate(input_batch, target_batch)
    gt = np.vstack(gt)
    scores = model.results_val["scores"]
    ids = [
        os.path.splitext(os.path.basename(x))[0].split("_embedded")[0]
        for x in dataloader.dataset.files
    ]
    outputs_dict = {
        "gt": gt,
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


def main(model_path=None, w=False, rm_duplicates=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)
    args = model.args
    table = pd.read_csv(args.target_path)
    if rm_duplicates:  # Allows the use of upsampling.
        table = table.drop_duplicates()
    args.train = False
    data = Dataset_handler(args)
    dataloader = data.get_loader(training=False)
    results = test(model, dataloader, table)
    return results
