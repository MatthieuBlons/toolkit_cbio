import sys
import traceback
from abc import abstractmethod
import torch
import os
from slide.utils import get_weights_path
from torchinfo import summary as model_summary

"""
For the most part this file is a fork of https://github.com/mahmoodlab/TRIDENT/blob/main/trident/patch_encoder_models/load.py 
It contains an assortment of pretrained patch encoders, all loadable via the encoder_factory() function.
"""
SLIDE_TO_TILE_ENCODER_MATCH = {
    "threads": "conch_v15",
    "titan": "conch_v15",
    "prism": "virchow",
    "chief": "ctranspath",
    "gigapath": "prov_gigapath",
    "madeleine": "conch_v1",
}


def encoder_factory(model_name, pretrained=True, freeze=True, **kwargs):
    """
    Build a slide encoder model.

    Args:
        model_name (str): Name of the model to build.
        pretrained (bool): Whether to load pretrained weights.
        freeze (bool): Whether to freeze the weights of the model.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        torch.nn.Module: The slide encoder model.
    """

    if model_name.startswith("mean-"):
        enc = MeanSlideEncoder
        return enc(model_name=model_name)
    elif model_name == "titan":
        enc = TitanSlideEncoder
    elif model_name == "prism":
        enc = PRISMSlideEncoder
    elif model_name == "chief":
        enc = CHIEFSlideEncoder
    elif model_name == "gigapath":
        enc = GigaPathSlideEncoder
    elif model_name == "madeleine":
        enc = MadeleineSlideEncoder
    else:
        raise ValueError(f"Model type {model_name} not supported")

    return enc(pretrained=pretrained, freeze=freeze, **kwargs)


class BaseSlideEncoder(torch.nn.Module):

    def __init__(self, freeze=True, **build_kwargs):
        """
        Parent class for all pretrained slide encoders.
        """
        super().__init__()
        self.enc_name = None
        self.model, self.precision, self.embedding_dim = self._build(**build_kwargs)

        # Set all parameters to be non-trainable
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def print_summary(self, depth=4, verbose=1):
        model_summary(self.model, depth=depth, verbose=verbose)

    def forward(self, batch):
        """
        Can be overwritten if model requires special forward pass.
        """
        z = self.model(batch)
        return z

    @abstractmethod
    def _build(self, **build_kwargs):
        """
        Initialization method, must be defined in child class.
        """
        pass


# Add a class for attention mil slide representation
# class AttMILSlideEncoder(BaseSlideEncoder):
# not implemented yet


class PRISMSlideEncoder(BaseSlideEncoder):

    def _build(self, pretrained=True):

        self.enc_name = "prism"

        if sys.version_info < (3, 10):
            raise RuntimeError(
                "PRISM requires Python 3.10 or above. Please update your Python interpreter."
            )

        try:
            import environs  # weird dependencies required by PRISM
            import sacremoses
            from transformers import AutoModel, AutoConfig
        except:
            traceback.print_exc()
            raise Exception(
                "Please run `pip install environs==11.0.0 transformers==4.42.4 sacremoses==0.1.1` "
                "and ensure Python version is 3.10 or above."
            )

        if pretrained:
            model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True)
        else:
            model = AutoModel.from_config(AutoConfig.from_pretrained("paige-ai/Prism"))
        model.text_decoder = None
        precision = torch.float16
        embedding_dim = 1280
        return model, precision, embedding_dim

    def forward(self, batch, device="cuda"):
        # input should be of shape (batch_size, tile_seq_len, tile_embed_dim)
        # how tiles' positions are assessed
        x = batch["features"].to(device)
        z = self.model.slide_representations(x)
        z = z["image_embedding"]
        return z


class CHIEFSlideEncoder(BaseSlideEncoder):

    def _build(self, pretrained=True):

        self.enc_name = "chief"
        weights_path = get_weights_path("slide", self.enc_name)

        # Ensure model can be built.
        try:
            sys.path.append(weights_path)
            from models.CHIEF import CHIEF
        except Exception:
            traceback.print_exc()
            raise Exception(
                f"\nError: Unable to import the CHIEF repository from '{weights_path}'.\n\n"
                "To resolve this issue:\n"
                "1. Ensure you have cloned the CHIEF repository to a convenient location:\n"
                "   `git clone https://github.com/hms-dbmi/CHIEF/`\n"
                "2. Set the path to CHIEF repo in `trident/slide_encoder_models/load_ckpts.json`, e.g., `./CHIEF`.\n"
                "3. Verify that CHIEF dependencies are installed:\n"
                "   `pip install addict`\n\n"
            )

        # Ensure weights can be loaded.
        try:
            current_wd = os.getcwd()  # Get current working directory
            os.chdir(weights_path)  # Change to CHIEF repo directory
            os.makedirs(os.path.join(weights_path, "model_weight"), exist_ok=True)

            required_files = {
                "Text_emdding.pth": "https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV",
                "CHIEF_pretraining.pth": "https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV",
            }

            for file_name, download_link in required_files.items():
                file_path = os.path.join(weights_path, "model_weight", file_name)
                if not os.path.exists(file_path):
                    raise Exception(
                        f"\nError: Missing required file '{file_name}'.\n\n"
                        "To resolve this issue:\n"
                        f"1. Download the file from:\n   {download_link}\n"
                        f"2. Copy '{file_name}' to the following directory:\n   {file_path}\n\n"
                        "Ensure the file is correctly placed before retrying."
                    )

            print("All necessary files are present. CHIEF setup is complete!")

        except Exception as e:
            print("\nAn error occurred during CHIEF setup:")
            traceback.print_exc()
            raise e

        model = CHIEF(size_arg="small", dropout=True, n_classes=2)

        # Load pretrained weights
        if pretrained:
            td = torch.load(
                os.path.join("model_weight", "CHIEF_pretraining.pth"),
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(td, strict=True)

        # Return to original working directory
        os.chdir(current_wd)

        precision = torch.float32
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch, device="cuda"):
        # how tiles' positions are assessed
        x = batch["features"].squeeze(0).to(device)
        z = self.model(x, torch.tensor([0]))
        z = z["WSI_feature"]  # Shape (1,768)
        return z


class GigaPathSlideEncoder(BaseSlideEncoder):

    def _build(self, pretrained=True):

        self.enc_name = "gigapath"

        try:
            from gigapath.slide_encoder import create_model
        except:
            traceback.print_exc()
            raise Exception(
                "Please install fairscale and gigapath using `pip install fairscale git+https://github.com/prov-gigapath/prov-gigapath.git`."
            )

        # Make sure flash_attn is correct version
        try:
            import flash_attn

            assert flash_attn.__version__ == "2.5.8"
        except:
            traceback.print_exc()
            raise Exception(
                "Please install flash_attn version 2.5.8 using `pip install flash_attn==2.5.8`."
            )

        if pretrained:
            model = create_model(
                "hf_hub:prov-gigapath/prov-gigapath",
                "gigapath_slide_enc12l768d",
                1536,
                global_pool=True,
            )
        else:
            model = create_model(
                "", "gigapath_slide_enc12l768d", 1536, global_pool=True
            )

        precision = torch.float16
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch, device="cuda"):
        self.model.tile_size = batch["attributes"]["patch_size_level0"]
        z = self.model(
            batch["features"].to(device),
            batch["coords"].to(device),
            all_layer_embed=True,
        )[11]
        return z


class MadeleineSlideEncoder(BaseSlideEncoder):

    def _build(self, pretrained=True):

        assert (
            pretrained
        ), "MadeleineSlideEncoder has no non-pretrained models. Please load with pretrained=True."

        self.enc_name = "madeleine"
        weights_path = get_weights_path("slide", self.enc_name)
        embedding_dim = 512

        try:
            from madeleine.models.factory import create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception(
                "Please install Madeleine using `pip install git+https://github.com/mahmoodlab/MADELEINE.git`"
            )

        model, precision = create_model_from_pretrained(weights_path)

        return model, precision, embedding_dim

    def forward(self, x, device="cuda"):
        # how tiles' positions are assessed
        z = self.model.encode_he(x["features"], device)
        return z


class TitanSlideEncoder(BaseSlideEncoder):

    def _build(self, pretrained=True):
        self.enc_name = "titan"
        assert (
            pretrained
        ), "TitanSlideEncoder has no non-pretrained models. Please load with pretrained=True."
        from transformers import AutoModel

        model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        precision = torch.float16
        embedding_dim = 768
        return model, precision, embedding_dim

    def forward(self, batch, device="cuda"):
        # do I have to pass coords at level 0 ?
        z = self.model.encode_slide_from_patch_features(
            batch["features"].to(device),
            batch["coords"].to(device),
            batch["attributes"]["level_patch_size"],
        )
        return z


class MeanSlideEncoder(BaseSlideEncoder):

    def _build(self, model_name="mean-default"):
        self.enc_name = model_name

        if model_name == "mean-conch_v1":
            embedding_dim = 768
        elif model_name == "mean-conch_v15":
            embedding_dim = 768
        elif model_name == "mean-uni_v1":
            embedding_dim = 1024
        elif model_name == "mean-uni_v2":
            embedding_dim = 1536
        elif model_name == "mean-ctranspath":
            embedding_dim = 768
        elif model_name == "mean-phikon":
            embedding_dim = 768
        elif model_name == "mean-resnet50":
            embedding_dim = 1024
        elif model_name == "mean-gigapath":
            embedding_dim = 1536
        elif model_name == "mean-virchow":
            embedding_dim = 2560
        elif model_name == "mean-virchow2":
            embedding_dim = 2560
        elif model_name == "mean-hoptimus0":
            embedding_dim = 1536
        elif model_name == "mean-phikon_v2":
            embedding_dim = 1024
        elif model_name == "mean-musk":
            embedding_dim = 1024
        elif model_name == "mean-hibou_l":
            embedding_dim = 1024
        elif model_name == "mean-kaiko-vit8s":
            embedding_dim = 384
        elif model_name == "mean-kaiko-vit16s":
            embedding_dim = 384
        elif model_name == "mean-kaiko-vit8b":
            embedding_dim = 768
        elif model_name == "mean-kaiko-vit16b":
            embedding_dim = 768
        elif model_name == "mean-kaiko-vit14l":
            embedding_dim = 1024
        elif model_name == "lunit-vits8":
            embedding_dim = 384
        else:
            print(
                f"WARNING: Could not automatically infer embedding_dim for mean encoder {self.enc_name}. Setting to None."
            )
            embedding_dim = None
        return None, None, embedding_dim

    def forward(self, batch, device="cuda"):
        z = batch["features"].to(device).mean(dim=1)  # Just mean pooling
        return z


def fetch_embeddings_dim(tile_encoder):
    embedding_dim = None
    if tile_encoder == "conch_v1":
        embedding_dim = 768
    elif tile_encoder == "conch_v15":
        embedding_dim = 768
    elif tile_encoder == "uni_v1":
        embedding_dim = 1024
    elif tile_encoder == "uni_v2":
        embedding_dim = 1536
    elif tile_encoder == "ctranspath":
        embedding_dim = 768
    elif tile_encoder == "phikon":
        embedding_dim = 768
    elif tile_encoder == "resnet50":
        embedding_dim = 1024
    elif tile_encoder == "gigapath":
        embedding_dim = 1536
    elif tile_encoder == "virchow":
        embedding_dim = 2560
    elif tile_encoder == "virchow2":
        embedding_dim = 2560
    elif tile_encoder == "hoptimus0":
        embedding_dim = 1536
    elif tile_encoder == "phikon_v2":
        embedding_dim = 1024
    elif tile_encoder == "musk":
        embedding_dim = 1024
    elif tile_encoder == "hibou_l":
        embedding_dim = 1024
    elif tile_encoder == "kaiko-vit8s":
        embedding_dim = 384
    elif tile_encoder == "kaiko-vit16s":
        embedding_dim = 384
    elif tile_encoder == "kaiko-vit8b":
        embedding_dim = 768
    elif tile_encoder == "kaiko-vit16b":
        embedding_dim = 768
    elif tile_encoder == "kaiko-vit14l":
        embedding_dim = 1024
    elif tile_encoder == "lunit-vits8":
        embedding_dim = 384
    else:
        print(
            f"WARNING: Tile embedding dimension is not known a priori for {tile_encoder}. Setting to None."
        )
    return embedding_dim
