import torch
import torch.nn as nn
import torch.nn.functional as F


class Apply2DTform(nn.Module):
    """
    Spatial Transformer layer implementation as described in [1].

    The layer is composed of 4 elements:

    -   _Imwarp: takes the input 2D image Batch <Img> of size (B, H, W, 1),
        the transformation vector <Tform> of size (B, 6), the output size
        <output_size> of a sample Img in the batch (H*, W*, 1), the padding
        method <padding> and, the interpolation method <interp>
        and outputs the warped output Batch of size (B, H*, W*, 1)

    -   _meshgrid: generates a grid of (x, y) coordinates, wth regards to the
        transformation, that correspond to a set of points where the
        input should be sampled to produce the transformed output.

    -   _interpolate: takes as input the original image, the grid (x, y)
        and, and produces the output transformed image using the intrepolation
        method

    -   _get_pixel_value: takes an image as input and outputs pixels value at
        (x, y) coordinates.


    Input
    -----
    -   input_size<tuple>: size of the input image (H*, W*, 1)
    -   output_size<tuple>: size of the output image (H*, W*, 1)
    -   Img<tensor or array>: the input 2D image batch of size (B, H, W, 1)
    -   Tform<tensor or array>: transformations to apply to each input sample (B, 6)
        Initialize to identity matrix.
    -   padding<bool>: apply padding before interpolation True/False
    -   interp<str>: interpolation method; supported_interp = ['Bilinear','Nearest']

    Returns
    -------
    -   out: transformed input image batch. Tensor of size (B, H*, W*, 1).

    Use
    ---

    out = Apply2DTform(input_size, output_size)(Img, Tform, padding, interp)

    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)
    """

    def __init__(self, input_size, output_size):
        super(Apply2DTform, self).__init__()
        self.output_size = output_size
        self.input_size = input_size

    def forward(self, Img, Tform, padding=False, interp="Bilinear"):
        supported_interp = ["Bilinear", "Nearest"]
        if interp not in supported_interp:
            raise ValueError(
                "Supported interp methods: 'Bilinear' (Default) or 'Nearest'"
            )

        output = self._Imwarp(
            Img, Tform, self.input_size, self.output_size, padding, interp
        )
        return output

    def _interpolate(self, Img, x_s, y_s, method):
        pad = (0, 0, 0, 1, 0, 1)  # Pad right and bottom
        Img = F.pad(Img, pad)

        H = Img.shape[1]
        W = Img.shape[2]

        max_x = H - 1
        max_y = W - 1

        x = 0.5 * (x_s + 1) * (max_x - 1)
        y = 0.5 * (y_s + 1) * (max_y - 1)

        x0 = torch.round(x).to(torch.int32)
        x1 = x0 + 1
        y0 = torch.round(y).to(torch.int32)
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        I00 = self._get_pixel_value(Img, x0, y0)
        I01 = self._get_pixel_value(Img, x0, y1)
        I10 = self._get_pixel_value(Img, x1, y0)
        I11 = self._get_pixel_value(Img, x1, y1)

        x0, x1 = x0.float(), x1.float()
        y0, y1 = y0.float(), y1.float()

        if method == "Bilinear":
            W00 = (x1 - x) * (y1 - y)
            W11 = (x - x0) * (y - y0)
            W01 = (x1 - x) * (y - y0)
            W10 = (x - x0) * (y1 - y)
        elif method == "Nearest":
            W00 = ((x - x0) < (x1 - x)).float() * ((y - y0) < (y1 - y)).float()
            W11 = ((x1 - x) < (x - x0)).float() * ((y1 - y) < (y - y0)).float()
            W01 = ((x - x0) < (x1 - x)).float() * ((y1 - y) < (y - y0)).float()
            W10 = ((x1 - x) < (x - x0)).float() * ((y - y0) < (y1 - y)).float()

        W00, W11, W01, W10 = (
            W00.unsqueeze(-1),
            W11.unsqueeze(-1),
            W01.unsqueeze(-1),
            W10.unsqueeze(-1),
        )

        out = W00 * I00 + W01 * I01 + W10 * I10 + W11 * I11
        return out

    def _meshgrid(self, height, width, vector, matrix):
        batch_size = vector.shape[0]
        ax, ay = torch.linspace(-1.0, 1.0, height), torch.linspace(-1.0, 1.0, width)
        x_t, y_t = torch.meshgrid(ax, ay, indexing="ij")

        x_t = x_t.flatten().unsqueeze(0)
        y_t = y_t.flatten().unsqueeze(0)
        sampling_grid = (
            torch.cat([x_t, y_t], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
        )

        vector = vector.unsqueeze(-1).repeat(1, 1, height * width)
        matrix = matrix.float()
        sampling_grid = sampling_grid.float()

        batch_grids = (matrix @ sampling_grid) + vector
        batch_grids = batch_grids.view(batch_size, 2, height, width)
        return batch_grids

    def _get_pixel_value(self, Img, x, y):
        B, H, W = x.shape
        batch_idx = torch.arange(0, B, dtype=torch.int64).view(B, 1, 1).expand(B, H, W)
        indices = torch.stack([batch_idx, x, y], dim=-1)
        return Img[indices[..., 0], indices[..., 1], indices[..., 2]]

    def _Imwarp(self, Img, Tform, input_size, output_size, padding, interp):
        B, H, W = Img.shape[:3]
        V = Tform[:, 4:].view(B, 2).float()
        M = Tform[:, :4].view(B, 2, 2).float()

        indx_grid = self._meshgrid(output_size[0], output_size[1], V, M)
        x_s, y_s = indx_grid[:, 0, :, :], indx_grid[:, 1, :, :]

        if padding:
            H, W = input_size
            top_pad = int((output_size[0] - H) / 2)
            bottom_pad = output_size[0] - H - top_pad
            left_pad = int((output_size[1] - W) / 2)
            right_pad = output_size[1] - W - left_pad
            Img = F.pad(Img, (0, 0, left_pad, right_pad, top_pad, bottom_pad))

        output_Img = self._interpolate(Img, x_s, y_s, method=interp)
        return output_Img


def Make_2Dtform(dim=(512, 512), trans=(0, 0), rot=0, scale=(0, 0), shear=(0, 0)):
    """
    Make_2Dtform: generates a 2D transformation matrix with specific parameters.

    Input
    -----
    -   dim <tuple>: Dimension of image to transform
    -   trans <tuple>: translation (in voxel)
    -   rot <float>: rotation (in degrees)
    -   scale <tuple>: scaling
    -   shear <tuple>: shear

    Returns
    -------
    -   tform: Tensor of size (1, 6).

    Use
    ---
    tform = Make_2Dtform(dim=(512, 512),
                         trans=(0,0),
                         rot=0,
                         scale=(0, 0),
                         shear=(0, 0))
    """
    # Translation
    trans = (
        torch.tensor(trans, dtype=torch.float32)
        / torch.tensor(dim, dtype=torch.float32)
        * 2
    )
    tx = trans[0]
    ty = trans[1]
    # Rotation
    rz = rot * (torch.pi / 180)  # Convert degrees to radians
    cos = torch.tensor([1, 0, 0, 1], dtype=torch.float32).unsqueeze(0) * torch.cos(
        torch.tensor(rz)
    )
    sin = torch.tensor([0, -1, 1, 0], dtype=torch.float32).unsqueeze(0) * torch.sin(
        torch.tensor(rz)
    )
    rot_matrix = cos + sin
    # Scaling
    scx, scy = scale[0], scale[1]
    scalex = torch.tensor([1, 0, 0, 0], dtype=torch.float32).unsqueeze(0) * scx
    scaley = torch.tensor([0, 0, 0, 1], dtype=torch.float32).unsqueeze(0) * scy
    scale_matrix = (
        torch.tensor([1, 0, 0, 1], dtype=torch.float32).unsqueeze(0) + scalex + scaley
    )
    # Shear
    shx, shy = shear[0], shear[1]
    shearx = torch.tensor([0, 1, 0, 0], dtype=torch.float32).unsqueeze(0) * shx
    sheary = torch.tensor([0, 0, 1, 0], dtype=torch.float32).unsqueeze(0) * shy
    shear_matrix = (
        torch.tensor([1, 0, 0, 1], dtype=torch.float32).unsqueeze(0) + shearx + sheary
    )
    # Combine transformations
    matrix = torch.matmul(
        shear_matrix.view(2, 2),
        torch.matmul(scale_matrix.view(2, 2), rot_matrix.view(2, 2)),
    )
    tform = torch.cat([matrix.view(1, 4), trans.unsqueeze(0)], dim=-1)

    return tform


class Apply2DDispField(torch.nn.Module):
    """
    Spatial Transformer layer implementation for 2D displacement field.

    The layer applies a 2D displacement field to a 2D image using the specified
    padding and interpolation methods.

    Input
    -----
    - input_size <tuple>: size of the input image (H*, W*, 1)
    - output_size <tuple>: size of the output image (H*, W*, 1)
    - Img <tensor>: the input 2D image batch of size (B, H, W, 1)
    - DispField <tensor>: displacement field to apply to each input sample (B, H*, W*, 2)
    - padding <bool>: apply padding before interpolation True/False
    - interp <str>: interpolation method; supported_interp = ['Bilinear','Nearest']

    Returns
    -------
    - out: transformed input image batch. Tensor of size (B, H*, W*, 1).
    """

    def __init__(self, input_size, output_size):
        super(Apply2DDispField, self).__init__()
        self.output_size = output_size
        self.input_size = input_size

    def forward(self, Img, DispField, padding=False, interp="Bilinear"):
        supported_interp = ["Bilinear", "Nearest"]
        if interp not in supported_interp:
            raise ValueError(
                "Supported interp keywords: 'Bilinear' (Default) or 'Nearest'"
            )
        output = self._Imwarp(
            Img, DispField, self.input_size, self.output_size, padding, interp
        )
        return output

    def _meshgrid(self, height, width, displacement):
        batch_size = displacement.shape[0]
        ax, ay = torch.linspace(-1.0, 1.0, height), torch.linspace(-1.0, 1.0, width)
        x_t, y_t = torch.meshgrid(ax, ay, indexing="ij")
        x_t, y_t = x_t.flatten(), y_t.flatten()
        sampling_grid = (
            torch.stack([x_t, y_t], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
        )

        dx, dy = displacement[:, :, :, 0].flatten(start_dim=1), displacement[
            :, :, :, 1
        ].flatten(start_dim=1)
        d_grid = torch.stack([dx, dy], dim=1)
        batch_grids = sampling_grid - d_grid
        return batch_grids.view(batch_size, 2, height, width)

    def _interpolate(self, Img, x_s, y_s, method):
        pad = (0, 0, 0, 1, 0, 1)  # Pad right and bottom
        Img = F.pad(Img, pad)

        H = Img.shape[1]
        W = Img.shape[2]

        max_x, max_y = H - 1, W - 1
        x, y = 0.5 * (x_s + 1) * (max_x - 1), 0.5 * (y_s + 1) * (max_y - 1)

        x0, x1 = x.floor().long(), (x.floor() + 1).long()
        y0, y1 = y.floor().long(), (y.floor() + 1).long()

        x0, x1 = x0.clamp(0, max_x), x1.clamp(0, max_x)
        y0, y1 = y0.clamp(0, max_y), y1.clamp(0, max_y)

        I00 = self._get_pixel_value(Img, x0, y0)
        I01 = self._get_pixel_value(Img, x0, y1)
        I10 = self._get_pixel_value(Img, x1, y0)
        I11 = self._get_pixel_value(Img, x1, y1)

        if method == "Bilinear":
            W00 = (x1 - x) * (y1 - y)
            W11 = (x - x0) * (y - y0)
            W01 = (x1 - x) * (y - y0)
            W10 = (x - x0) * (y1 - y)
        else:  # Nearest neighbor
            W00 = ((x - x0) < (x1 - x)).float() * ((y - y0) < (y1 - y)).float()
            W11 = ((x1 - x) < (x - x0)).float() * ((y1 - y) < (y - y0)).float()
            W01 = ((x - x0) < (x1 - x)).float() * ((y1 - y) < (y - y0)).float()
            W10 = ((x1 - x) < (x - x0)).float() * ((y - y0) < (y1 - y)).float()

        W00, W01, W10, W11 = (
            W00.unsqueeze(-1),
            W01.unsqueeze(-1),
            W10.unsqueeze(-1),
            W11.unsqueeze(-1),
        )
        out = W00 * I00 + W01 * I01 + W10 * I10 + W11 * I11
        return out

    def _get_pixel_value(self, Img, x, y):
        B, H, W = x.size(0), x.size(1), x.size(2)
        batch_idx = torch.arange(B).view(B, 1, 1).repeat(1, H, W)
        indices = torch.stack([batch_idx, x, y], dim=3)
        return Img[indices[:, :, :, 0], indices[:, :, :, 1], indices[:, :, :, 2]]

    def _Imwarp(self, Img, DispField, input_size, output_size, padding, interp):
        height, width = output_size
        indx_grid = self._meshgrid(height, width, DispField)
        x_s, y_s = indx_grid[:, 0, :, :], indx_grid[:, 1, :, :]

        if padding:
            H, W = input_size
            top_pad = int((output_size[0] - H) / 2)
            bottom_pad = output_size[0] - H - top_pad
            left_pad = int((output_size[1] - W) / 2)
            right_pad = output_size[1] - W - left_pad
            Img = F.pad(Img, (0, 0, left_pad, right_pad, top_pad, bottom_pad))

        output_Img = self._interpolate(Img, x_s, y_s, method=interp)
        return output_Img
