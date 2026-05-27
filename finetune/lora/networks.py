"""
Implements the networks that can be used in train.
use of pytorch.
"""

import functools
from slide.tile_encoder.load import encoder_factory
from torch.nn import (
    Linear,
    Module,
    Sequential,
    Identity,
    ReLU,
    Dropout,
    BatchNorm1d,
    BatchNorm2d,
)
from mil.deepmil.utils import is_in_args
from torchinfo import summary
from peft import get_peft_model
import torch


def get_norm_layer(use_bn=True, d=1):
    bn_dict = {1: BatchNorm1d, 2: BatchNorm2d}
    if use_bn:  # Use batch
        norm_layer = functools.partial(
            bn_dict[d], affine=True, track_running_stats=True
        )
    else:
        norm_layer = functools.partial(Identity)
    return norm_layer


class Linear_bn(Module):
    def __init__(self, in_channels, out_channels, dropout, use_bn):
        self.norm_layer = get_norm_layer(use_bn)
        super(Linear_bn, self).__init__()
        self.layer = Sequential(
            Linear(in_features=in_channels, out_features=out_channels),
            self.norm_layer(out_channels),
            ReLU(),
            Dropout(p=dropout),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class MLP(Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.dropout = is_in_args(args, "dropout", 0.5)

        self.width_fe = is_in_args(args, "width_fe", 64)

        self.feature_dim = is_in_args(args, "feature_dim", 512)
        self.feature_depth = is_in_args(args, "feature_depth", 512)

        self.output_lenght = is_in_args(args, "output_lenght", 16)
        self.n_layers_classif = is_in_args(args, "n_layers_classif", 1)

        self.transform = self.instance_transf()
        self.classifier = self.mlp_classifier()

    def instance_transf(self):
        transform = Sequential(
            Linear(self.feature_dim, self.feature_depth),
            ReLU(),
            Dropout(p=self.dropout),
        )
        return transform

    def mlp_classifier(self):
        classifier = []
        classifier.append(
            Linear_bn(
                int(self.feature_depth),
                self.width_fe,
                self.dropout,
                use_bn=True,
            )
        )
        for i in range(self.n_layers_classif):
            classifier.append(
                Linear_bn(
                    self.width_fe,
                    self.width_fe,
                    self.dropout,
                    use_bn=True,
                )
            )
        classifier.append(Linear(self.width_fe, self.output_lenght))
        classifier.append(ReLU())
        return Sequential(*classifier)

    def forward(self, x):
        """
        Input x of size F where :
            * F is the dimension of feature space
        """
        (
            bs,
            _,
        ) = x.shape
        # change -> name conflict self.args.instance_transf vs self.instance_transf
        if self.args.instance_transf:
            x = self.transform(x)
        out = self.classifier(x)
        out = out.view((bs, self.output_lenght))
        return out


# switch lora on/off
class LoRAFactory(Module):
    """
    LoRAFactory.
    This framework makes things easy if we want to plug the
    LoRA framework on a foundation feature extractor (taking images as input).
    """

    def __init__(self, args, peft_cfg, lora=True):
        super(LoRAFactory, self).__init__()
        self.args = args
        self.cfg = peft_cfg
        self.lora = lora
        self.precision = {}
        self.backbone, self.transform = self.get_backbone()
        self.apply_lora()
        self.head = self.get_head()
        if args.device == "cpu":
            self.cast_backbone(precision=torch.float32)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def get_backbone(self):
        pretrained = encoder_factory(self.args.encoder)
        backbone, transform, precision = (
            pretrained.model,
            pretrained.eval_transforms,
            pretrained.precision,
        )
        backbone = backbone.to(precision)
        self.precision["backbone"] = precision
        return backbone, transform

    def get_head(self, precision=torch.float32):
        head = MLP(self.args)
        head = head.to(precision)
        self.precision["head"] = precision
        return head

    def apply_lora(self, precision=torch.float32):
        if self.lora:
            # turn lora on
            self.backbone = get_peft_model(self.backbone, self.cfg)
            for name, module in self.backbone.named_modules():
                if "lora_" in name:
                    module.to(dtype=precision)
            self.precision["lora"] = precision

    def cast_backbone(self, precision):
        self.backbone.to(precision)
        self.precision["backbone"] = precision

    def print_lora_summary(self):
        if self.lora:
            self.backbone.print_trainable_parameters()
        else:
            print(f"LoRA is disable")

    def print_summary(self, model="full", depth=4, verbose=1):
        if model == "full":
            summary(Sequential(self.backbone, self.head), depth=depth, verbose=verbose)
        elif model == "backbone":
            summary(self.backbone, depth=depth, verbose=verbose)
        elif model == "head":
            summary(self.head, depth=depth, verbose=verbose)
        else:
            ValueError
