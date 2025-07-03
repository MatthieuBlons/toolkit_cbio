"""
Implements the networks that can be used in train.
use of pytorch.
"""

import functools
from torch.nn import (
    Linear,
    Module,
    Sequential,
    Softmax,
    Identity,
    Conv1d,
    Conv2d,
    ReLU,
    Dropout,
    BatchNorm1d,
    BatchNorm2d,
    InstanceNorm1d,
    LogSoftmax,
)
from torch.nn.parameter import Parameter
import torch
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
from mil.deepmil.utils import is_in_args
from torchinfo import summary


def get_norm_layer(use_bn=True, d=1):
    bn_dict = {1: BatchNorm1d, 2: BatchNorm2d}
    if use_bn:  # Use batch
        norm_layer = functools.partial(
            bn_dict[d], affine=True, track_running_stats=True
        )
    else:
        norm_layer = functools.partial(Identity)
    return norm_layer


class Linear_norm(Module):
    def __init__(
        self, in_features, out_features, dropout, constant_size, dim_batch=None
    ):
        if dim_batch is None:
            dim_batch = out_features
        super(Linear_norm, self).__init__()
        self.cs = constant_size
        self.block = Sequential(
            Linear(in_features, out_features),
            ReLU(),  # Added 25/09
            Dropout(p=dropout),  # Added 25/09
            self.get_norm(constant_size, dim_batch),
        )

    def get_norm(self, constant_size, out_features):
        if not constant_size:
            norm = InstanceNorm1d(out_features)
        else:
            norm = BatchNorm1d(out_features)
        return norm

    def forward(self, x):
        x = self.block(x)
        return x


class Conv2d_bn(Module):
    def __init__(self, in_channels, out_channels, dropout, use_bn):
        super(Conv2d_bn, self).__init__()
        self.norm_layer = get_norm_layer(use_bn, d=2)
        self.layer = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            self.norm_layer(out_channels),
            ReLU(),
            Dropout(p=dropout),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class Conv1d_bn(Module):
    def __init__(self, in_channels, out_channels, dropout, use_bn):
        self.norm_layer = get_norm_layer(use_bn)
        super(Conv1d_bn, self).__init__()
        self.layer = Sequential(
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            self.norm_layer(out_channels),
            ReLU(),
            Dropout(p=dropout),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


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


class PoolingFunction(Module):
    def __init__(self, args):
        super(PoolingFunction, self).__init__()
        self.pooling = args.pooling_fct
        self.args = args
        self.k = args.k
        if self.pooling in ["attn", "attn_max"]:
            self.attention = Sequential(MultiHeadAttention(args), Softmax(dim=-2))
        elif self.pooling in ["attn_gated"]:
            self.attention = Sequential(MultiHeadGatedAttention(args), Softmax(dim=-2))

    def forward(self, x):
        if self.pooling in ["attn", "attn_gated"]:
            w = self.attention(x)  # (bs, nbt, nheads)
            w = torch.transpose(w, -1, -2)  # (bs, nheads, nbt)
            slide = torch.matmul(
                w, x
            )  # Slide representation, shape (bs, nheads, nfeatures)
            slide = slide.flatten(1, -1)  # (bs, nheads*nfeatures)
        elif self.pooling == "mean":
            slide = torch.mean(x, -2)  # BxF
        elif self.pooling == "max":
            slide, _ = torch.max(x, dim=-2)  # BxF
        elif self.pooling == "attn_max":
            w = self.attention(x)
            _, inds = torch.max(w, dim=-2)
            slide = torch.gather(
                x, 1, torch.cat([inds.unsqueeze(-1)] * self.args.feature_depth, axis=-1)
            )
            slide = slide.squeeze(-2)
        else:
            print(
                f"WARNING: {self.pooling} pooling function not yet implemented. Will use mean pooling."
            )
            slide = torch.mean(x, -2)  # BxF
        return slide


class MultiHeadAttention(Module):
    """
    Implements the multihead attention mechanism used in
    MultiHeadedAttentionMIL_*.
    Input (batch, nb_tiles, features)
    Output (batch, nb_tiles, nheads)
    """

    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        atn_dim = is_in_args(args, "atn_dim", 256)
        self.num_heads = is_in_args(args, "num_heads", 1)
        self.dropout = args.dropout
        self.dim_heads = atn_dim // self.num_heads
        assert (
            self.dim_heads * self.num_heads == atn_dim
        ), "atn_dim must be divisible by num_heads"

        self.atn_layer_1_weights = Parameter(torch.Tensor(atn_dim, args.feature_depth))
        self.atn_layer_2_weights = Parameter(
            torch.Tensor(1, 1, self.num_heads, self.dim_heads, 1)
        )
        self.atn_layer_1_bias = Parameter(torch.empty((atn_dim)))
        self.atn_layer_2_bias = Parameter(torch.empty((1, self.num_heads, 1, 1)))
        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.atn_layer_1_weights)
        xavier_uniform_(self.atn_layer_2_weights)
        constant_(self.atn_layer_1_bias, 0)
        constant_(self.atn_layer_2_bias, 0)

    def forward(self, x):
        """Extracts a series of attention scores.

        Args:
            x (torch.Tensor): size (batch, nb_tiles, features)

        Returns:
            torch.Tensor: size (batch, nb_tiles, nb_heads)
        """
        bs, nbt, _ = x.shape

        # Weights extraction
        x = F.linear(x, weight=self.atn_layer_1_weights, bias=self.atn_layer_1_bias)
        x = torch.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view((bs, nbt, self.num_heads, 1, self.dim_heads))
        x = torch.matmul(x, self.atn_layer_2_weights) + self.atn_layer_2_bias  # scores.
        x = x.view(bs, nbt, -1)  # shape (bs, nbt, nheads)
        return x


class MultiHeadGatedAttention(Module):
    """
    Implements the multihead gated attention mechanism.
    Input (batch, nb_tiles, features)
    Output (batch, nb_tiles, nheads)
    """

    def __init__(self, args):
        super(MultiHeadGatedAttention, self).__init__()
        atn_dim = is_in_args(args, "atn_dim", 256)
        self.num_heads = is_in_args(args, "num_heads", 1)
        self.dropout = args.dropout
        self.dim_heads = atn_dim // self.num_heads
        assert (
            self.dim_heads * self.num_heads == atn_dim
        ), "atn_dim must be divisible by num_heads"

        self.V = Parameter(torch.Tensor(atn_dim, args.feature_depth))
        self.V_bias = Parameter(torch.empty((atn_dim)))
        self.U = Parameter(torch.Tensor(atn_dim, args.feature_depth))
        self.U_bias = Parameter(torch.empty((atn_dim)))
        self.W = Parameter(torch.Tensor(1, 1, self.num_heads, self.dim_heads, 1))
        self.W_bias = Parameter(torch.empty((1, self.num_heads, 1, 1)))
        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.U)
        xavier_uniform_(self.V)
        xavier_uniform_(self.W)
        constant_(self.U_bias, 0)
        constant_(self.V_bias, 0)
        constant_(self.W_bias, 0)

    def forward(self, x):
        """Extracts a series of attention scores.

        Args:
            x (torch.Tensor): size (batch, nb_tiles, features)

        Returns:
            torch.Tensor: size (batch, nb_tiles, nb_heads)
        """
        bs, nbt, _ = x.shape
        # Weights extraction
        v = F.linear(x, weight=self.V, bias=self.V_bias)
        v = torch.tanh(v)
        u = F.linear(x, weight=self.U, bias=self.U_bias)
        u = torch.sigmoid(u)
        x = v * u
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view((bs, nbt, self.num_heads, 1, self.dim_heads))
        x = torch.matmul(x, self.W) + self.W_bias  # scores.
        x = x.view(bs, nbt, -1)  # shape (bs, nbt, nheads)
        return x


class MLP(Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.width_fe = is_in_args(args, "width_fe", 64)
        self.feature_dim = is_in_args(args, "feature_dim", 512)
        self.feature_depth = is_in_args(args, "feature_depth", 512)
        self.num_class = is_in_args(args, "num_class", 2)
        self.n_layers_classif = is_in_args(args, "n_layers_classif", 1)

        self.transform = self.instance_transf()
        self.classifier = self.mlp_classifier()

    def instance_transf(self):
        # Do we want batch norm on the enc?
        # transform = Linear_norm(self.feature_dim,self.feature_depth,self.dropout,self.args.constant_size,)
        transform = Sequential(
            Linear(self.feature_dim, self.feature_depth),
            ReLU(),
            Dropout(p=self.dropout),
        )
        return transform

    def mlp_classifier(self):
        classifier = []
        classifier.append(
            Linear_norm(
                int(self.feature_depth),
                self.width_fe,
                self.dropout,
                self.args.constant_size,
            )
        )
        for i in range(self.n_layers_classif):
            classifier.append(
                Linear_norm(
                    self.width_fe, self.width_fe, self.dropout, self.args.constant_size
                )
            )
        classifier.append(Linear(self.width_fe, self.num_class))
        classifier.append(LogSoftmax(-1))
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
        if self.args.instance_transf:
            x = self.transform(x)
        if not self.args.constant_size:
            x = x.unsqueeze(-2)
        out = self.classifier(x)
        out = out.view((bs, self.num_class))
        return out


class MHMClayers(Module):
    """
    MultiHeadMultiClass attention MIL, with several layers in the decision MLP.
    Same as MultiHeadedAttentionMIL_multiclass but have a classifier with N
    Linear layers.
    N is parametrized by args by args.n_layers_classif
    """

    def __init__(self, args):
        super(MHMClayers, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.width_fe = is_in_args(args, "width_fe", 64)
        self.atn_dim = is_in_args(args, "atn_dim", 256)
        self.feature_dim = is_in_args(args, "feature_dim", 512)
        self.feature_depth = is_in_args(args, "feature_depth", 512)
        self.num_heads = is_in_args(args, "num_heads", 1)
        self.num_class = is_in_args(args, "num_class", 2)
        self.n_layers_classif = is_in_args(args, "n_layers_classif", 1)
        self.dim_heads = self.atn_dim // self.num_heads
        assert (
            self.dim_heads * self.num_heads == self.atn_dim
        ), "atn_dim must be divisible by num_heads"

        # make a function
        # do I want to normalise on args.n_tiles when args.constant_size
        # self.instance_transf = Linear_norm(self.feature_dim, self.feature_depth, self.dropout, args.constant_size,)
        self.instance_transf = Sequential(
            Linear(self.feature_dim, self.feature_depth),
            ReLU(),
            Dropout(p=self.dropout),
        )

        self.pooling_function = PoolingFunction(self.args)

        # make a function
        classifier = []
        classifier.append(
            Linear_norm(
                int(self.feature_depth * self.num_heads),
                self.width_fe,
                self.dropout,
                args.constant_size,
            )
        )
        for i in range(self.n_layers_classif):
            classifier.append(
                Linear_norm(
                    self.width_fe, self.width_fe, self.dropout, args.constant_size
                )
            )
        classifier.append(Linear(self.width_fe, self.num_class))
        classifier.append(LogSoftmax(-1))
        self.classifier = Sequential(*classifier)

    def forward(self, x):
        """
        Input x of size NxF where :
            * F is the dimension of feature space
            * N is number of patche
        """
        bs, _, _ = x.shape
        if self.args.instance_transf:
            x = self.instance_transf(x)
        slide = self.pooling_function(x)
        if not self.args.constant_size:
            slide = slide.unsqueeze(-2)
        out = self.classifier(slide)
        out = out.view((bs, self.num_class))
        return out


class MILFactory(Module):
    """
    MILFactory.
    This framework makes things easy if we want to plug the
    MIL framework on a learnable feature extractor (taking images as input).
    Not implemented yet (only takes feat vectors as input not images)
    """

    def __init__(self, args):
        super(MILFactory, self).__init__()
        self.args = args
        self.name, self.mil = self.get_model(args)

    def forward(self, x):
        if self.args.wsi_enc == "tile":
            if self.args.constant_size:
                batch_size, nb_tiles = x.shape[0], x.shape[1]
            else:
                batch_size, nb_tiles = 1, x.shape[-2]
            x = x.view(batch_size, nb_tiles, self.args.feature_dim)
        elif self.args.wsi_enc == "slide":
            batch_size = x.shape[0]
            x = x.view(batch_size, self.args.feature_dim)
        x = self.mil(x)
        return x

    def get_model(self, args):
        model_name = args.model
        # tmp fix to use model train with old model_name
        if model_name == "mhmc" or model_name == "mhmclayers":
            mil = MHMClayers
        # Possibility to had model
        elif model_name == "mlp":
            mil = MLP
        else:
            raise ValueError(f"Unknown encoder name {model_name}")
        return model_name, mil(args)

    def print_summary(self, depth=4, verbose=1):
        summary(self.mil, depth=depth, verbose=verbose)
