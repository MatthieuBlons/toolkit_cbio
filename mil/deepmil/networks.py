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
    GELU,
    SiLU,
    Dropout,
    LayerNorm,
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
import math


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
            GELU(),  # Added 25/09
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
            GELU(),
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
        self.dropout = is_in_args(args, "dropout", 0.1)
        self.feature_dim = is_in_args(args, "feature_dim", 1536)
        self.feature_depth = is_in_args(args, "feature_depth", 512)
        self.width_fe = is_in_args(args, "width_fe", 512)
        self.n_layers_classif = is_in_args(args, "n_layers_classif", 0)
        self.output_layer = is_in_args(args, "output_layer", None)
        self.num_class = is_in_args(args, "num_class", 2)
        self.transform = self.instance_transf()
        self.classifier = self.mlp_classifier()

    def instance_transf(self):
        transform = Linear_norm(
            in_features=self.feature_dim,
            out_features=self.feature_depth,
            dropout=0.1,
            constant_size=self.args.constant_size,
        )
        return transform

    def mlp_classifier(self):
        classifier = []
        if self.n_layers_classif > 0:
            classifier.append(
                Linear_norm(
                    self.feature_depth,
                    self.width_fe,
                    self.dropout,
                    self.args.constant_size,
                )
            )
            for i in range(self.n_layers_classif - 1):
                classifier.append(
                    Linear_norm(
                        self.width_fe,
                        self.width_fe,
                        self.dropout,
                        self.args.constant_size,
                    )
                )
            classifier.append(Linear(self.width_fe, self.num_class))
        else:
            classifier.append(Linear(self.feature_depth, self.num_class))

        if self.output_layer == "pseudo_proba":
            classifier.append(LogSoftmax(-1))
        elif self.output_layer == "proba":
            classifier.append(Softmax(-1))

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
    MultiHeadMultiClass attention MIL,
    with Linear layers in pre- and post-attention modules.
    """

    def __init__(self, args):
        super(MHMClayers, self).__init__()
        self.args = args
        self.dropout = is_in_args(args, "dropout", 0.25)
        self.feature_dim = is_in_args(args, "feature_dim", 1536)
        self.feature_depth = is_in_args(args, "feature_depth", 512)
        self.atn_dim = is_in_args(args, "atn_dim", 256)
        self.num_heads = is_in_args(args, "num_heads", 1)
        self.width_fe = is_in_args(args, "width_fe", 512)
        self.n_layers_classif = is_in_args(args, "n_layers_classif", 1)
        self.output_layer = is_in_args(args, "output_layer", "logsoftmax")
        self.num_class = is_in_args(args, "num_class", 1)
        self.dim_heads = self.atn_dim // self.num_heads
        assert (
            self.dim_heads * self.num_heads == self.atn_dim
        ), "atn_dim must be divisible by num_heads"

        # pre-attention
        self.instance_transf = None
        # one layer linear projection
        if self.args.instance_transf == "linear":
            self.instance_transf = Linear_norm(
                in_features=self.feature_dim,
                out_features=self.feature_depth,
                dropout=0.1,
                constant_size=args.constant_size,
            )
        # one layer linear projection + one transfomer block
        if self.args.instance_transf == "transformer":
            assert (
                self.feature_dim == self.feature_depth
            ), "when using transformer layer make sure feature_depth == feature_dim"
            self.instance_transf = TransLayer(
                hidden_dim=self.feature_depth, num_heads=1, mlp_dim=512, drop=0.1
            )
        # one layer linear projection + one transfomer block + Rope
        if self.args.instance_transf == "roformer":
            assert (
                self.feature_dim == self.feature_depth
            ), "when using transformer layer make sure feature_depth == feature_dim"
            self.instance_transf = TransLayer(
                hidden_dim=self.feature_depth,
                num_heads=1,
                mlp_dim=512,
                drop=0.1,
                rope=True,
            )
        # one layer linear projection + one transfomer block + Relative Position bias
        if self.args.instance_transf == "rposbias":
            assert (
                self.feature_dim == self.feature_depth
            ), "when using transformer layer make sure feature_depth == feature_dim"
            self.instance_transf = TransLayer(
                hidden_dim=self.feature_depth,
                num_heads=1,
                mlp_dim=512,
                drop=0.1,
                rpb=True,
            )

        # attention
        self.pooling_function = PoolingFunction(self.args)

        # post-attention
        classifier = []
        if self.n_layers_classif > 0:
            classifier.append(
                Linear_norm(
                    int(self.feature_depth * self.num_heads),
                    self.width_fe,
                    0.1,
                    args.constant_size,
                )
            )
            for i in range(self.n_layers_classif - 1):
                classifier.append(
                    Linear_norm(self.width_fe, self.width_fe, 0.1, args.constant_size)
                )
            classifier.append(Linear(self.width_fe, self.num_class))
        else:
            classifier.append(
                Linear(int(self.feature_depth * self.num_heads), self.num_class)
            )

        # output
        if self.output_layer == "logsoftmax":
            classifier.append(LogSoftmax(-1))
        elif self.output_layer == "softmax":
            classifier.append(Softmax(-1))
        self.classifier = Sequential(*classifier)

    def forward(self, x):
        """
        Input x of size NxF where :
            * F is the dimension of feature space
            * N is number of patche
        """

        bs, _, _ = x.shape
        # Patch transformation
        if self.instance_transf is not None:
            if (self.args.instance_transf == "roformer"
                or self.args.instance_transf == "rposbias"
            ):
                x, coords = x
                x = self.instance_transf(x, coords)
            else:
                x = self.instance_transf(x)

        # Attention
        slide = self.pooling_function(x)
        if not self.args.constant_size:
            slide = slide.unsqueeze(-2)

        # Classifier
        out = self.classifier(slide)

        # Output layer
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
        if self.args.constant_size:
            batch_size, nb_tiles = x.shape[0], x.shape[1]
        else:
            batch_size, nb_tiles = 1, x.shape[-2]
        x = x.view(batch_size, nb_tiles, self.args.feature_dim)
        x = self.mil(x)
        return x

    def get_model(self, args):
        model_name = args.model
        if model_name == "mhmc":
            mil = MHMClayers
        # Possibility to had model
        else:
            raise ValueError(f"Unknown encoder name {model_name}")
        return model_name, mil(args)

    def print_summary(self, depth=4, verbose=1):
        summary(self.mil, depth=depth, verbose=verbose)


# transformers related:
class SinCosEncoding(Module):
    def __init__(
        self,
        feature_dim,
        dim=2,
        freq=10000,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.dim = dim
        assert self.feature_dim % self.dim == 0, print(
            f"SinCosEncoding requires // {self.dim}"
        )
        self.freq = freq

    def sincos(self, pos, feature_dim, freq):
        """
        pos: (B, N)
        returns: (B, N, dim)
        """
        device = pos.device
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2, device=device)
            * (-math.log(freq) / feature_dim)
        )  # (dim/2,)
        pos = pos.unsqueeze(-1)  # (B, N, 2)
        pe = torch.zeros(*pos.shape[:-1], feature_dim, device=device)  # (B, N, dim)
        pe[..., 0::2] = torch.sin(pos * div_term)
        pe[..., 1::2] = torch.cos(pos * div_term)
        return pe

    def forward(self, x, pos):
        """
        x: (B, N, D)
        pos: (B, N, dim)  (pixel positions)
        """
        if self.dim == 2:
            x_pos = pos[..., 0]  # (B, N)
            y_pos = pos[..., 1]  # (B, N)
            pe_x = self.sincos(x_pos, self.feature_dim // 2, self.freq)
            pe_y = self.sincos(y_pos, self.feature_dim // 2, self.freq)
            pe = torch.cat([pe_x, pe_y], dim=-1)  # (B, N, D)
            return x + pe
        else:
            pe = self.sincos(pos, self.feature_dim, self.freq)  # (B, N, D)
            return x + pe


class LearnedPosEncoding(Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=128,
    ):
        """
        dim: embedding dimension
        hidden_dim: size of hidden layer in MLP
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = Sequential(
            Linear(2, hidden_dim), GELU(), Linear(hidden_dim, feature_dim)
        )

    def forward(self, x, pos):
        """
        x: (B, N, D)
        pos: (B, N, 2)  (pixel coordinates)
        """
        pe = self.mlp(pos)  # (B, N, D)
        return x + pe


class PosEncoding(Module):
    pos_encoding_mapping = {
        "sincos": SinCosEncoding,
        "learned": LearnedPosEncoding,
        # "relative": RelativePosEncoding,
    }

    def __init__(self, feature_dim, strategy=None, *args, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.strategy = strategy

        if self.strategy is not None:
            assert feature_dim % 2 == 0, "PosEncoding requires even dimension"
            self.position = self._get_encoding_strategy(self.strategy, *args, **kwargs)

    def _get_encoding_strategy(self, enc_strategy: str, *args, **kwargs) -> callable:
        if enc_strategy.lower() in PosEncoding.pos_encoding_mapping.keys():
            return PosEncoding.pos_encoding_mapping[enc_strategy.lower()](
                *args,
                **kwargs,
            )
        # Otherwise raise an error
        else:
            raise NotImplementedError(
                "pos encoding strategy: {} is not implemented.".format(enc_strategy)
                + f"\nPlease choose a valid strategy from: {' ,'.join(PosEncoding.pos_encoding_mapping.keys())}."
            )

    def forward(self, x, coords=None):
        if self.strategy is None:
            return x

        if coords is None:
            raise ValueError("Coords are required for positional encoding")

        return self.position(x, coords)


class RoPE(Module):
    def __init__(self, feature_dim, freq=10000):
        """ """
        super().__init__()
        assert feature_dim % 2 == 0, "RoPE requires even dimension"

        self.feature_dim = feature_dim
        self.freq = freq

        inv_freq = 1.0 / (
            freq ** (torch.arange(0, feature_dim, 2).float() / feature_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def _get_angles(self, coords):
        """ """
        # simple 2D → 1D projection (sum works well in practice)
        pos = coords[..., 0] + coords[..., 1]  # (B, N)

        freqs = torch.einsum("bn,d->bnd", pos, self.inv_freq)
        return freqs

    def _rotate_half(self, x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, q, k, coords):
        """ """
        freqs = self._get_angles(coords)

        cos = torch.cos(freqs).unsqueeze(1)
        sin = torch.sin(freqs).unsqueeze(1)

        # expand to full dim
        cos = torch.repeat_interleave(cos, 2, dim=-1)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot


class RelativePositionBias(Module):
    def __init__(self, num_heads, hidden_dim=128):
        super().__init__()
        self.num_heads = num_heads

        self.mlp = Sequential(
            Linear(2, hidden_dim), GELU(), Linear(hidden_dim, num_heads)
        )

    def forward(self, coords):
        """
        coords: (B, N, 2)

        returns:
            bias: (B, num_heads, N, N)
        """
        B, N, _ = coords.shape

        # Compute pairwise relative positions
        dist = coords[:, :, None, :] - coords[:, None, :, :]  # (B, N, N, 2)

        # Flatten for MLP
        dist_flat = dist.view(B * N * N, 2)

        bias = self.mlp(dist_flat)  # (B*N*N, num_heads)

        bias = bias.view(B, N, N, self.num_heads)
        bias = bias.permute(0, 3, 1, 2)  # (B, H, N, N)

        return bias


class GatedMLP(Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()

        # project to 2 * hidden_dim (for gating)
        self.fc1 = Linear(dim, hidden_dim * 2)

        self.act = SiLU()

        self.fc2 = Linear(hidden_dim, dim)
        self.drop = Dropout(drop)

    def forward(self, x):
        x_proj = self.fc1(x)  # (B, N, 2 * hidden_dim)

        x, gate = x_proj.chunk(2, dim=-1)  # split

        x = x * self.act(gate)  # gating

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class TransLayer(Module):
    def __init__(
        self,
        hidden_dim,
        num_heads=1,
        mlp_dim=2048,
        drop=0,
        attn_drop=0,
        rope=False,
        rpb=False,
        scale=True,
        ls_init=1e-5,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Self-Attention
        self.norm1 = LayerNorm(hidden_dim)
        self.qkv = Linear(hidden_dim, hidden_dim * 3)
        self.attn_drop = Dropout(attn_drop)
        self.proj = Linear(hidden_dim, hidden_dim)
        self.proj_drop = Dropout(drop)
        self.ls1 = Parameter(ls_init * torch.ones(hidden_dim)) if scale else None

        # Rope if needed
        self.rope = RoPE(self.head_dim) if rope else None

        # Or Relative Position Bias
        self.rpb = RelativePositionBias(num_heads) if rpb else None

        # Gated MLP
        self.norm2 = LayerNorm(hidden_dim)
        self.mlp = GatedMLP(hidden_dim, mlp_dim, drop)
        self.ls2 = Parameter(ls_init * torch.ones(hidden_dim)) if scale else None

    def forward(self, x, coords=None):
        B, N, D = x.shape
        # Attention
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)  # (B, H, N, D_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k, coords)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)

        if self.rpb is not None:
            bias = self.rpb(coords)  # (B, H, N, N)
            attn = attn + bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)

        out = self.proj(out)
        out = self.proj_drop(out)

        if self.ls1 is not None:
            x = x + self.ls1 * out
        else:
            x = x + out

        # MLP
        mlp_out = self.mlp(self.norm2(x))

        if self.ls2 is not None:
            x = x + self.ls2 * mlp_out
        else:
            x = x + mlp_out

        return x
