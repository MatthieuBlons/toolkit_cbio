import numpy as np
from skimage import transform
from imregister.transform import Apply2DTform, Make_2Dtform
import torch
import torch.nn as nn
from imregister.similarity import MutualInformation


class registration(nn.Module):
    def __init__(self, fov, max_iteration=100, pyramidal_level=3, method="translation"):
        super(registration, self).__init__()
        self.fov = fov
        self.max_iteration = max_iteration  # max iteration
        # initial transformation (identity by default)
        self.init_tform = self._get_ientity(self.fov)
        self.metric = self._get_metric()
        self.radius = 1e-2  # initial perturbation radius in fraction of the image
        self.max_radius = 1  #  max step length reduction factor
        self.min_radius = 1e-6  #  max step length reduction factor
        self.relaxation = 0.02  # Step length reduction factor
        self.tolerance = 1e-3  # min difference between old and new metric score to consider no provement
        self.best_score = None
        self.best_tform = None
        self.pyramidal_level = pyramidal_level
        self.update_tform = getattr(self, method + "_update")

    def _get_ientity(self, fov):
        identity = Make_2Dtform(dim=fov)
        return identity

    def _get_metric(self):
        metric = MutualInformation(num_bins=128, sigma=0.4, normalize=True)
        return metric

    def _compute_metrics(self, x, y):
        metric_score = self.metric(x, y)
        return metric_score

    def _keep_best_tform(self, tfrom, score, level):
        if self.best_score is None:
            self.best_score = self.info_dict["score"][level][-1]
            self.best_tform = self.info_dict["transform"][level][-1]
        if score > self.best_score:
            self.best_tform = tfrom
            self.best_score = score

    def _get_transformer(self, dim):
        affine = Apply2DTform(input_size=dim, output_size=dim)
        return affine

    def _in_fov(self, img, numpy=False):
        img_size = img.shape
        in_tensor = torch.tensor(img.reshape((1, *img_size, 1)))
        self.identity = self._get_ientity(img_size)
        out = Apply2DTform(input_size=img_size, output_size=self.fov)(
            in_tensor, self.identity, padding=True, interp="Bilinear"
        )
        if numpy:
            out = out.numpy().squeeze()
        return out

    def translation_update(self, tform, radius):
        # B = Batch
        # Generate small random perturbations for translation only
        current_matrix = tform[:, :4].view(2, 2)

        # Translation perturbation
        translation_shift = (2 * radius) * torch.rand(1, 2) - radius
        current_translation = tform[:, 4:].view(1, 2)
        new_translation = current_translation + translation_shift

        # Combine rotation and translation into a new transformation matrix
        new_tform = torch.cat(
            [current_matrix.view(1, 4), new_translation.view(1, 2)], dim=-1
        )
        return new_tform
        return 0

    def rigid_update(self, tform, radius):
        # Generate small random perturbations for rigid transformation rotation and translation only

        # Rotation perturbation (sould I use the same perturbation radius?)
        theta_shift = 2 * radius * torch.rand(1) - radius  # Random angle in radians
        cos_theta = torch.cos(theta_shift)
        sin_theta = torch.sin(theta_shift)
        rotation_shift = torch.tensor(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
        ).squeeze()
        current_rotation = tform[:, :4].view(2, 2)
        new_rotation = torch.matmul(rotation_shift, current_rotation)

        # Translation perturbation
        translation_shift = (2 * radius) * torch.rand(1, 2) - radius
        current_translation = tform[:, 4:].view(1, 2)
        new_translation = current_translation + translation_shift

        # Combine rotation and translation into a new rigid transformation
        new_tform = torch.cat(
            [new_rotation.view(1, 4), new_translation.view(1, 2)], dim=-1
        )
        return new_tform

    def affine_update(self, tform, radius):
        # Generate small random perturbations for affine transformation

        # matrix perturbation (sould I use the same perturbation radius?)
        # sould I decompose the matrix in rot, scale, shear
        # sould I allow shear ?

        # small rotation perturbation
        theta = 2 * radius * torch.rand(1) - radius
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotation = torch.tensor(
            [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
        ).squeeze()

        # small scale perturbations

        # use another radius
        scale_factors = 1 + (2 * 0.01 * torch.rand(2) - 0.01)
        scale = torch.diag(scale_factors)
        # make scale homogene in both direction or allows scalex scaley
        # scale = torch.diag(torch.tensor((scale_factors[0], scale_factors[0])))

        # small shear perturbations
        shear_factors = 2 * radius * torch.rand(2) - radius
        shear_factors = torch.tensor((0, 0), dtype=torch.float32)
        shear = torch.tensor([[1, shear_factors[0]], [shear_factors[1], 1]])

        # Combine transformations: shear -> scale -> rotation -> current
        current_matrix = tform[:, :4].view(2, 2)
        new_matrix = torch.matmul(
            torch.matmul(torch.matmul(shear, scale), rotation), current_matrix
        )

        # Translation perturbation
        translation_shift = (2 * radius) * torch.rand(1, 2) - radius
        current_translation = tform[:, 4:].view(1, 2)
        new_translation = current_translation + translation_shift

        # Combine rotation and translation into a new rigid transformation
        new_tform = torch.cat(
            [new_matrix.view(1, 4), new_translation.view(1, 2)], dim=-1
        )
        return new_tform

    def _increase_radius(self, radius):
        if (radius > self.min_radius) & (radius <= self.max_radius):
            # print("perturabtion change: +", radius * self.relaxation)
            new = radius + (radius * self.relaxation)
        else:
            new = radius
        return new

    def _decrease_radius(self, radius):
        if (radius > self.min_radius) & (radius <= self.max_radius):
            # print("perturabtion change: -", radius * self.relaxation)
            new = radius - (radius * self.relaxation)
        else:
            new = radius
        return new

    def _get_pyramidal(self, origin, level, numpy=True):
        scale = 1 / (2**level)
        ipyramid = transform.rescale(origin, scale)
        size = ipyramid.shape
        if not numpy:
            ipyramid = torch.tensor(ipyramid.reshape((1, *size, 1)))
        return ipyramid, scale, size

    def _init_dict(self):
        self.info_dict = {
            "pyramidal_level": self.pyramidal_level,
            "max_iteration": self.max_iteration,
            "transform": [[] for l in range(self.pyramidal_level + 1)],
            "score": [[] for l in range(self.pyramidal_level + 1)],
            "radius": [[] for l in range(self.pyramidal_level + 1)],
            "mov": [],
            "fix": [],
            "reg": [],
            "interm": [],
        }

    def _update_dict(self, tform, score, radius, level=0):
        self.info_dict["transform"][level].append(tform)
        self.info_dict["score"][level].append(score)
        self.info_dict["radius"][level].append(radius)

    def _flush_init_tform(self):
        self.best_score = None
        self.init_tform = self.best_tform

    def _InitRadiusOnPyramid(self, size, nsize=(2, 2), frac=0.1):
        nb_pixels = map(lambda size: np.ceil(size * frac), size)
        res = map(lambda nsize, size: nsize / size, nsize, size)
        radius = min(nb_pixels) * min(res)
        self.radius = radius
        return radius

    def _image_registration(self, mov, fix, level):
        assert mov.shape == fix.shape
        self.AFFINE = self._get_transformer(fix.squeeze().shape)
        for i in range(self.max_iteration):
            tform = self.info_dict["transform"][level][-1]
            radius = self.info_dict["radius"][level][-1]
            if i == 0:
                interm = self.AFFINE(mov, tform, padding=True, interp="Bilinear")
                self.info_dict["interm"].append(interm)

            new_tform = self.update_tform(tform, radius)
            temp = self.AFFINE(mov, new_tform, padding=True, interp="Bilinear")

            score = self._compute_metrics(temp, fix)

            if score >= self.info_dict["score"][level][-1]:
                new_radius = self._increase_radius(radius)
                self._update_dict(new_tform, score, new_radius, level)
                self._keep_best_tform(tform, score, level)
            else:
                new_radius = self._decrease_radius(radius)
                self._update_dict(tform, score, new_radius, level)

        reg = self.AFFINE(mov, self.best_tform, padding=True, interp="Bilinear")
        return reg

    def forward(self, mov, fix):
        perturbation = 0.1
        mov = self._in_fov(mov, numpy=True)
        fix = self._in_fov(fix, numpy=True)
        assert mov.shape == fix.shape
        self._init_dict()
        for level in range(self.pyramidal_level, -1, -1):
            temp_mov, scale_mov, size_mov = self._get_pyramidal(
                mov, level=level, numpy=False
            )

            temp_fix, scale_fix, size_fix = self._get_pyramidal(
                fix, level=level, numpy=False
            )
            assert temp_mov.shape == temp_fix.shape
            print(f"level = {level}, mov  size = {size_mov}, downscale = {scale_mov}")
            print(f"level = {level}, fix size = {size_fix}, downscale = {scale_fix}")

            tform = self.init_tform
            print(f"level = {level}, perturbation frac = {perturbation}")
            radius = self._InitRadiusOnPyramid(size_mov, frac=perturbation)
            score = self._compute_metrics(temp_mov, temp_fix)
            self._update_dict(tform, score, radius, level)

            print(f"level = {level}, init score = {self.info_dict["score"][level][-1]}")
            print(
                f"level = {level}, init tform = {self.info_dict["transform"][level][-1]}"
            )
            print(
                f"level = {level}, init radius = {self.info_dict["radius"][level][-1]}"
            )

            temp_reg = self._image_registration(temp_mov, temp_fix, level=level)

            print(f"level = {level}, best score = {self.best_score}")
            print(f"level = {level}, best tform = {self.best_tform}")

            self.info_dict["mov"].append(temp_mov)
            self.info_dict["fix"].append(temp_fix)
            self.info_dict["reg"].append(temp_reg)

            self._flush_init_tform()
            perturbation *= 0.2

        return temp_reg, self.best_tform
