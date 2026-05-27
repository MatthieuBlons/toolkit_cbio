import torch
from finetune.lora.models import load_model_from_path
from finetune.lora.utils import print_dict
from finetune.lora.dataloader import Dataset_handler
from tqdm import tqdm
import numpy as np
import h5py
from PIL import Image


class ModelHooker:
    """
    Manage the hooks for the LoRA Model.
    """

    def __init__(self, network):
        self.place_hooks(network)

    def _get_encodings_hook(self):
        def hook_fundation(m, i, o):
            """
            Hooks the output of fundation model.
            """
            tile_encodings = i[0]
            self.tile_encodings = tile_encodings.detach().cpu().numpy()

        return hook_fundation

    def _get_instance_transf_hook(self):
        def hook_instance_transf(m, i, o):
            """
            Hooks the output of instance transformation layers.
            """
            tile_transf = o[0]
            self.tile_transf = tile_transf.detach().cpu().numpy()

        return hook_instance_transf

    def _get_tile_representation_hook(self):
        def hook_repre(m, i, o):
            """
            Hooks the antepenultiem layer of the MLP Head.
            """
            tile_rep = i[0]
            self.tile_rep = tile_rep.squeeze().detach().cpu().numpy()

        return hook_repre

    def _get_outputs_hook(self):
        def hook_output(m, i, o):
            """
            Hooks the outputs of the MLP Head.
            """
            output = o[0]
            self.output = output.detach().cpu().numpy()

        return hook_output

    def place_hooks(self, net):
        for name, layer in net.named_children():
            if list(layer.children()):
                self.place_hooks(layer)

            if name == "model":
                hook_layer = list(layer.children())[-1]
                hook_layer.register_forward_hook(self._get_encodings_hook())

            if name == "transform":
                hook_layer = list(layer.children())[0]
                hook_layer.register_forward_hook(self._get_instance_transf_hook())

            if name == "classifier":
                hook_rep = list(layer.children())[-2]
                hook_rep.register_forward_hook(self._get_tile_representation_hook())
                hook_out = list(layer.children())[-1]
                hook_out.register_forward_hook(self._get_outputs_hook())


def predict(model, features, dataloader):
    """
    prediction with model on image
    """
    model.network.eval()
    hooker = ModelHooker(model.network)
    progress = tqdm(
        desc=f"prediction...",
        total=len(dataloader),
        unit="batch",
        initial=0,
        leave=False,
    )
    pred = []
    emb = []
    for input_batch in dataloader:
        pred.append(model.predict(input_batch).squeeze())
        emb.append(np.array(hooker.tile_encodings.squeeze()))
        progress.update()
    pred = np.vstack(pred)
    emb = np.vstack(emb)

    img_paths = [x for x in dataloader.dataset.input_files]

    outputs_dict = {
        "img_path": np.array(img_paths),
        "embeddings": emb,
        "pred": pred,
        "features": features,
    }
    return outputs_dict


def preprocess(path, transform=None):
    img = np.asarray(Image.open(path))
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    if img.dtype not in [np.uint8, np.float32]:
        img = np.float32(img)

    img = Image.fromarray(img).convert("RGB")
    if transform is not None:
        img = transform(img)
    img = img.unsqueeze(0)
    return img


def predict_one_image(model, image_path):
    """
    prediction with model on image
    """
    model.network.eval()
    hooker = ModelHooker(model.network)
    image = preprocess(image_path, model.network.transform)

    print(f"image shape = {image.shape}")
    print(f"image type = {image.dtype}")

    pred = model.predict(image).squeeze()
    print(f"pred shape = {pred.shape}")
    print(f"pred type = {pred.dtype}")

    emb = np.array(hooker.tile_encodings.squeeze())
    print(f"emb shape = {emb.shape}")
    print(f"emb type = {emb.dtype}")

    outputs_dict = {
        "embeddings": emb,
        "pred": pred,
    }
    return outputs_dict


def main(
    model_path: str,
    img_dir: str,
    input_path: str | None = None,
    target_path: str | None = None,
    batch_size: int =1,
    device: str | None=None,
    verbose: int=0,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device used = {device}")
    model = load_model_from_path(model_path, device)

    if model.network.lora:
        model.network.print_lora_summary()
        model.network.print_summary(verbose=verbose)
    args = model.args

    if target_path is not None:
        training_target = target_path
    else:
        training_target = args.target_path
    with h5py.File(training_target, "r") as f:
        attrs = dict(f["signature"].attrs)
        features = attrs["feat"]

    args.train = False
    args.img_dir = img_dir
    args.input_path = input_path
    args.target_path = None
    args.batch_size = batch_size

    print_dict(args.__dict__, name="model args for pred")

    data = Dataset_handler(
        args, img_only=True, preprocess=model.network.transform, predict=True
    )
    dataloader = data.get_loader(training=False)

    results = predict(model, features, dataloader)
    return results
