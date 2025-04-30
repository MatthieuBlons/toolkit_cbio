"""
Just predicts, given a set of WSI.
"""

import numpy as np
from glob import glob
import torch
import pandas as pd
import os
from .models import DeepMIL
from sklearn.preprocessing import Normalizer


def load_model(model_path, device):
    """Loads and prepare a learned model for prediction.
    Args:
        model_path (str): path to the *.pt.tar model
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    args.device = device
    model = DeepMIL(
        args, label_encoder=checkpoint["label_encoder"], ipca=None
    )
    model.network.load_state_dict(checkpoint["state_dict"])
    model.network.eval()
    model.target_table = (
        checkpoint["target_table"]
        if "target_table" in checkpoint
        else model.args.target_table
    )
    return model


def preprocessing(wsi, device):
    norm = Normalizer()
    wsi = np.load(wsi)
    wsi = norm.fit_transform(wsi)
    wsi = torch.Tensor(wsi)
    wsi = wsi.unsqueeze(0)
    wsi = wsi.to(device).float()
    return wsi


def predict(model, wsi_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    wsis = glob(os.path.join(wsi_dir, "*.h5"))
    assert (
        wsis
    ), f"wsis have to be stored directly into {wsi_dir}, with a .h5 extension.x"
    results = {"name": [], "pred": [], "proba": []}
    for wsi in wsis:
        name = os.path.splitext(os.path.basename(wsi))[0]
        wsi = preprocessing(wsi, device)
        proba, y_hat = model.predict(wsi)
        results["name"].append(name)
        results["pred"].append(y_hat)
        results["proba"].append(proba)
    return results
