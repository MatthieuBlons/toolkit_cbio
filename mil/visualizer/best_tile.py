from mil.deepmil.predict import load_model
from matplotlib.colors import Normalize
from scipy.special import softmax
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from skimage import filters
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from mil.visualizer.model_hooker import HookerAttnMIL
from slide.utils import read_h5_coords, get_slide_reader
from slide.tile import SlidePatcher
import h5py
from osfile.manager import findFile, safe_mkdir
from draw.effect import blend_images
from scipy.ndimage import gaussian_filter
from slide.utils import get_x_y_to
from PIL import Image


class BaseTileVisualizer(ABC):
    def __init__(
        self,
        model: str,
        wsi_dir: str,
        feat_dir: str | None = None,
        target_path: str | None = None,
        device: str = "cpu",
    ):
        ## Model loading
        self.device = device
        self.model_path = model
        self.model = load_model(model, self.device)
        self.label_encoder = self.model.label_encoder
        # Par défaut on considère le dataset d'entrainement = données contenues dans args.
        self.table = (
            self._load_table(self.model.target_table)
            if target_path is None
            else self._load_table(target_path)
        )
        self.wsi_dir = wsi_dir
        self.feat_dir = self.model.args.wsi_dir if feat_dir is None else feat_dir
        self.model_name = self.model.network.name
        self.target_name = self.model.args.target_name
        self.num_class = self.model.args.num_class
        self.num_heads = self.model.args.num_heads
        self.instance_transf = self.model.network.mil.instance_transf
        self.instance_transf.eval()
        self.attention = self.model.network.mil.pooling_function
        self.attention.eval()
        self.classifier = self.model.network.mil.classifier
        self.classifier.eval()
        self.hooker = HookerAttnMIL(self.model.network, self.model.args.num_heads)

    def _load_table(self, table):
        if type(table) is str:
            table = pd.read_csv(table)
        return table

    def _get_info(self, ID):
        feat_path = os.path.join(self.feat_dir, f"{ID}.h5")
        attrs, coords = read_h5_coords(feat_path)
        return attrs, coords

    def _get_slide(self, ID):
        wsi_path, cnt = findFile(self.wsi_dir, ID, fileExtensions=False)
        assert wsi_path, f"no wsi with name {ID}"
        assert (
            cnt < 2
        ), f"several wsi with name: {ID} were found... Will open: {wsi_path[0]}"
        reader = get_slide_reader(wsi_path)
        slide = reader(img_path=wsi_path)
        return slide

    def _init_patcher(self, ID):
        attrs, coords = self._get_info(ID)
        slide = self._get_slide(ID)
        patcher = SlidePatcher(
            slide,
            mag_0=slide.magnification,
            mag_target=attrs["target_magnification"],
            patch_size=attrs["target_patch_size"],
            overlap=attrs["target_overlap"],
            custom_xywh=coords,
            mask_tolerance=attrs["tissu_thr"],
            xywh_only=False,
        )
        return patcher

    def _get_thumbnail(self, ID, size=(1024, 1024), numpy=True):
        slide = self._get_slide(ID)
        thumbnail = slide.get_thumbnail(size)
        if numpy:
            thumbnail = np.array(thumbnail)[:, :, :3]
        return thumbnail

    def _get_image(self, ID, indice):
        """_get_image.
        extract the indice-ieme tile of the wsi stored at path_raw.
        returns numpy array.
        :ID: name of the wsi (all extensions accepted)
        :param indice: number of the tile in the flattened WSI
        """
        patcher = self._init_patcher(ID)
        image = patcher.get_tile(*patcher.valid_patches[indice])
        return image

    def _get_embeddings(self, ID):
        # change attribute in tile patcher: "features" to -> "tile_features"
        feat_path = os.path.join(self.feat_dir, f"{ID}.h5")
        feat_key = "features"
        with h5py.File(feat_path, "r") as f:
            attrs = dict(f[feat_key].attrs)
            feats = f[feat_key][:]
        return attrs, feats

    @torch.inference_mode()
    def _preprocess(self, ID, expand_bs=True):
        """preprocess the input to feed the model
        Args:
            ID: name of the wsi (all extensions accepted)
        """
        _, feats = self._get_embeddings(ID)
        mat = feats[:, : self.model.args.feature_dim]
        mat = torch.from_numpy(mat).float()
        mat = mat.unsqueeze(0) if expand_bs else mat
        # Instance transform here ?
        mat = mat.to(self.device)
        if self.model.args.instance_transf:
            mat = self.instance_transf(mat)
        return mat

    @abstractmethod
    def forward(self, ID):
        pass


class TileSeeker(BaseTileVisualizer):
    """
    Decision-based extraction of predictive tiles.
    Takes as input a model of MIL (AttentionMIL), extracts the predictive tiles
    contained in the train and or test set.
    """

    def __init__(
        self,
        model: str,
        wsi_dir: str,
        feat_dir: str | None = None,
        target_path: str | None = None,
        device: str = "cpu",
        n_best: int = 1000,
        min_prob: bool = False,
        max_per_slides: int = 300,
        att_thres: str | int = "otsu",
        store: bool = False,
    ):
        """__init__.
        :param n_best: int, number of tiles to keep per class.
        :param min_prob: if True, keeps tiles MINIMIZING a given logit.
        :param max_per_slides: diversity parameter. A WSI can not participate to the selection with more than max_per_slides tiles.
        :param att_thres: 'str' or int: if str = 'otsu', use otsu threshold to select tiles, else, use setted number of tile threshold (usually 300)
        """
        super(TileSeeker, self).__init__(
            model=model,
            wsi_dir=wsi_dir,
            feat_dir=feat_dir,
            target_path=target_path,
            device=device,
        )
        self.n_best = n_best
        self.min_prob = min_prob
        self.max_per_slides = max_per_slides
        self.att_thres = att_thres
        self.store = store
        self.attention_scores = None
        self.decision_scores = None

        # Best tile for each head?
        assert (
            self.model.args.num_heads == 1
        ), "you can't extract a best tile when using the multiheaded attention"

        # lists that have to be filled with various WSI
        self._reset_storage()

    @torch.inference_mode()
    def forward(self, ID):
        """forward.
        Execute a forward pass through the MLP classifier.
        Stores the n-best tiles for each class.
        :param ID: name of the wsi as appearing in the table_data.
        """
        # process the wsi associated with ID (all tiles)
        attrs, coords = self._get_info(ID)

        # define the keys to remove
        keys = [
            key
            for key in attrs.keys()
            if key not in ["name", "level", "target_magnification"]
        ]
        for key in keys:
            attrs.pop(key, None)

        x = self._preprocess(ID)

        self.classifier(x)
        logits = self.hooker.scores.squeeze()

        self.attention(x)
        tw = self.hooker.tiles_weights.squeeze()

        # Best tiles on attention
        if self.att_thres == "otsu":
            otsu = filters.threshold_otsu(tw)
            selection = np.where(tw >= otsu)[0]
        elif isinstance(self.att_thres, int):
            _, ind = torch.sort(torch.Tensor(tw), dim=0)
            size_select = min(self.att_thres, len(ind))
            selection = ind[-size_select:].cpu().numpy()
        else:
            _, ind = torch.sort(torch.Tensor(tw), dim=0)
            selection = ind.cpu().numpy()

        if self.store:
            self.store_best(
                x.cpu().numpy().squeeze(),
                logits,
                coords,
                attrs,
                selection,
            )

    def forward_all(self):
        table = self.table
        ID = table["ID"].values
        for n, o in enumerate(ID):
            self.forward(o)

    def store_best(
        self,
        feats,
        logits,
        coords,
        attrs,
        selection,
    ):
        """store_best.
        decides if we have to store the tile, according the final activation value.
        :param out: out of a forward pass
        :param info: info dictionnary of the WSI
        :param min_prob: bool:
        maximiser proba -> prendre les derniers éléments de indices_best (plus grand au plus petit)
        minimiser proba -> prendre les premiers éléments de indice_best
        """
        sgn = -1 if self.min_prob else 1
        # for each tile
        (
            tmp_infos,
            tmp_coords,
            tmp_feats,
            tmp_attentions,
            tmp_preclassifs,
            tmp_scores,
            tmp_images,
        ) = (dict(), dict(), dict(), dict(), dict(), dict(), dict())

        for i, _ in enumerate(self.label_encoder.classes_):
            tmp_infos[i] = []
            tmp_coords[i] = []
            tmp_feats[i] = []
            tmp_attentions[i] = []
            tmp_preclassifs[i] = []
            tmp_scores[i] = []

            tmp_images[i] = []

        # patcher = self._init_patcher(attrs["name"])

        ## Selects the best tiles per WSI.
        for candidate in range(coords.shape[0]):
            if candidate not in selection:
                continue

            # for each class
            for i, _ in enumerate(self.label_encoder.classes_):
                # If the score for class o at tile s is bigger than the smallest
                # stored value: put in storage
                if (len(self.store_score[i]) < self.n_best) or (
                    sgn * logits[candidate, i] >= sgn * self.store_score[i][0]
                ):
                    tmp_infos[i].append(attrs)
                    tmp_coords[i].append(coords[candidate, :])
                    tmp_feats[i].append(feats[candidate, :])
                    tmp_attentions[i].append(self.hooker.tiles_weights[candidate])
                    tmp_preclassifs[i].append(self.hooker.reprewsi[candidate, :])
                    tmp_scores[i].append(logits[candidate, i])

                    # tmp_images[i].append(patcher.get_tile(*patcher.valid_patches[candidate]))

        # add the max_per_slides best tiles per WSI to storage.
        for i, _ in enumerate(self.label_encoder.classes_):
            selection = np.argsort(tmp_scores[i])[::-sgn][: self.max_per_slides]
            self.store_info[i] += list(np.array(tmp_infos[i])[selection])
            self.store_coords[i] += list(np.array(tmp_coords[i])[selection])
            self.store_feat[i] += list(np.array(tmp_feats[i])[selection])
            self.store_attention[i] += list(np.array(tmp_attentions[i])[selection])
            self.store_preclassif[i] += list(np.array(tmp_preclassifs[i])[selection])
            self.store_score[i] += list(np.array(tmp_scores[i])[selection])

            # self.store_image[i] += list(np.array(tmp_images[i])[selection])

        # Selects the n_best best tiles overall.
        for i, _ in enumerate(self.label_encoder.classes_):
            indices_best = np.argsort(self.store_score[i])[::sgn][-self.n_best :]
            self.store_info[i] = list(np.array(self.store_info[i])[indices_best])
            self.store_coords[i] = list(np.array(self.store_coords[i])[indices_best])
            self.store_feat[i] = list(np.array(self.store_feat[i])[indices_best])
            self.store_attention[i] = list(
                np.array(self.store_attention[i])[indices_best]
            )
            self.store_preclassif[i] = list(
                np.array(self.store_preclassif[i])[indices_best]
            )
            self.store_score[i] = list(np.array(self.store_score[i])[indices_best])

            # self.store_image[i] = list(np.array(self.store_image[i])[indices_best])

    def summarise_storage(self):
        # fetch a tile idx for easy identification
        summary = pd.DataFrame()
        for i, _ in enumerate(self.label_encoder.classes_):
            meta_df = pd.DataFrame(self.store_info[i])
            coords_df = pd.DataFrame(self.store_coords[i], columns=["x", "y", "w", "h"])
            merged_df = pd.merge(
                left=meta_df, right=coords_df, left_index=True, right_index=True
            )

            merged_df.insert(
                loc=0,
                column="img_path",
                value=merged_df.apply(
                    lambda row: "_".join(
                        [
                            row["name"],
                            str(row["x"]),
                            str(row["y"]),
                            str(row["w"]),
                            str(row["h"]),
                        ]
                    )
                    + ".jpeg",
                    axis=1,
                ),
            )
            merged_df["attention score"] = self.store_attention[i]
            merged_df["class label"] = i
            merged_df["class score"] = self.store_score[i]
            summary = pd.concat([summary, merged_df], axis=0)

        return summary

    def extract_images(self):
        # make it faster
        for i, _ in enumerate(self.label_encoder.classes_):
            assert self.store_score[i], f"no tile stored for class:{i}"
            for t in range(self.n_best):
                name = self.store_info[i][t]["name"]
                level = self.store_info[i][t]["level"]
                x, y, w, h = self.store_coords[i][t]
                slide = self._get_slide(name)
                tmp = slide.read_region(location=(x, y), level=level, size=(w, h))
                self.store_image[i].append(tmp)
        return self.store_image

    def _reset_storage(self):
        """_reset_storage.
        Reset the storage dict.
        store_score and store info are dict with keys the classes (ordinals)
        containing empty lists. When filled, they are supposed to n_best scores
        and infodicts values.
        only store images as the name of the targets as keys.
        Advice : fill store image at the end only.
        """
        self.store_info = dict()
        self.store_coords = dict()
        self.store_image = dict()
        self.store_feat = dict()
        self.store_attention = dict()
        self.store_preclassif = dict()
        self.store_score = dict()

        for i, _ in enumerate(self.label_encoder.classes_):
            self.store_info[i] = []
            self.store_coords[i] = []
            self.store_image[i] = []
            self.store_feat[i] = []
            self.store_attention[i] = []
            self.store_preclassif[i] = []
            self.store_score[i] = []


class ConsensusTileSeeker(TileSeeker):
    def __init__(
        self,
        model: str,
        wsi_dir: str,
        feat_dir: str | None = None,
        target_path: str | None = None,
        device: str = "cpu",
        n_best: int = 1000,
        min_prob: bool = False,
        max_per_slides: int = 300,
        att_thres: str | int = "otsu",
        store: bool = False,
    ):
        super(ConsensusTileSeeker, self).__init__(
            model[0],
            wsi_dir,
            feat_dir,
            target_path,
            device,
            n_best,
            min_prob,
            max_per_slides,
            att_thres,
            store,
        )

        tile_seekers = []
        for m in model:
            ts = TileSeeker(
                m,
                wsi_dir,
                feat_dir,
                target_path,
                device,
                n_best,
                min_prob,
                max_per_slides,
                att_thres,
                False,
            )
            tile_seekers.append(ts)
        self.seekers = tile_seekers

        ## list that have to be filled with various WSI
        self._reset_storage()

    @torch.inference_mode()
    def forward(self, ID):
        """forward.
        Execute a forward pass through the MLP classifier.
        Stores the n-best tiles for each class.
        :param wsi_ID: wsi_ID as appearing in the table_data.
        """
        # process the wsi associated with ID
        attrs, coords = self._get_info(ID)
        x = self._preprocess(ID)

        # define the keys to remove
        keys = [
            key
            for key in attrs.keys()
            if key not in ["name", "level", "target_magnification"]
        ]
        for key in keys:
            attrs.pop(key, None)

        outs = []
        lastrepr = []
        logits = []
        attention = []

        for s in self.seekers:
            try:
                test_fold = self.table.loc[self.table["ID"] == ID, "test"].values[0]
                if test_fold == s.model.args.test_fold:
                    outs.append(s.classifier(x).cpu().numpy())
                    logits.append(s.hooker.scores.squeeze())
                    lastrepr.append(s.hooker.reprewsi.squeeze())
                    s.attention(x)
                    attention.append(s.hooker.tiles_weights.squeeze())
            except:
                outs.append(s.classifier(x).cpu().numpy())
                logits.append(s.hooker.scores.squeeze())
                lastrepr.append(s.hooker.reprewsi.squeeze())
                s.attention(x)
                attention.append(s.hooker.tiles_weights.squeeze())

        out = np.mean(outs, 0)
        lastrepr = np.mean(lastrepr, 0)
        logits = np.mean(logits, 0)
        tw = np.mean(attention, 0)

        # filling the hooker with mean values
        self.hooker.tiles_weights = tw
        self.hooker.scores = logits
        self.hooker.reprewsi = lastrepr

        ## Otsu thresholding
        if self.att_thres == "otsu":
            otsu = filters.threshold_otsu(tw)
            selection = np.where(tw >= otsu)[0]
        elif isinstance(self.att_thres, int):
            _, ind = torch.sort(torch.Tensor(tw), dim=0)
            size_select = min(self.att_thres, len(ind))
            selection = ind[-size_select:].cpu().numpy()
        else:
            _, ind = torch.sort(torch.Tensor(tw), dim=0)
            selection = ind.cpu().numpy()

        # Find attention scores to filter out of distribution tiles
        if self.store:
            self.store_best(
                x.cpu().numpy().squeeze(),
                logits,
                coords,
                attrs,
                selection,
            )


class HeatmapMaker(BaseTileVisualizer):

    def __init__(
        self,
        model: str | list,
        wsi_dir: str,
        feat_dir: str | None = None,
        target_path: str | None = None,
        device: str = "cpu",
    ):

        super(HeatmapMaker, self).__init__(
            model[0],
            wsi_dir,
            feat_dir,
            target_path,
            device,
        )

        tile_seekers = []
        for m in model:
            ts = TileSeeker(
                m,
                wsi_dir,
                feat_dir,
                target_path,
                device,
                n_best=0,
                min_prob=False,
                max_per_slides=0,
                att_thres=0,
                store=False,
            )
            tile_seekers.append(ts)

        self.seekers = tile_seekers
        self.target_name = tile_seekers[0].model.args.target_name
        self.feat_dir = tile_seekers[0].feat_dir

    @torch.inference_mode()
    def forward(self, ID):
        """forward.
        Execute a forward pass through the MLP classifier.
        Stores the n-best tiles for each class.
        Args:
        Returns:
        """
        attrs, coords = self._get_info(ID)

        # process wsi
        x = self._preprocess(ID, expand_bs=False)
        outs = []
        logits = []
        attention = []
        for s in self.seekers:
            try:
                test_fold = self.table.loc[self.table["ID"] == ID, "test"].values[0]
                if test_fold == s.model.args.test_fold:
                    outs.append(s.classifier(x).cpu().numpy())
                    logits.append(s.hooker.scores.squeeze())
                    s.attention(x.unsqueeze(0))
                    attention.append(s.hooker.tiles_weights.squeeze())
            except:
                print(f"slide: {ID} not in test, will use all the models provided")
                outs.append(s.classifier(x).cpu().numpy())
                logits.append(s.hooker.scores.squeeze())
                s.attention(x.unsqueeze(0))
                attention.append(s.hooker.tiles_weights.squeeze())

        out = np.mean(outs, 0)
        logits = np.mean(logits, 0)
        tw = np.squeeze(np.mean(attention, 0))

        return tw, logits, attrs, coords

    def make_montage(
        self,
        ID,
        downsample: int = 1,
        smooth: int = 0,
        alpha: float = 0.5,
        save: str | None = None,
    ):
        """
        Generates and saves a heatmap overlay images for the given whole-slide image ID.
        Args:
        Returns:
        """
        overlays = {}
        slide = self._get_slide(ID)
        tw, logits, attrs, coords = self.forward(ID)
        (level_width, level_height) = slide.level_dimensions[attrs["level"]]
        thumbnail = self._get_thumbnail(
            ID, (level_width / downsample, level_height / downsample), numpy=True
        )
        thumbnail_height, thumbnail_width, _ = thumbnail.shape
        thumbnail_patch_size = max(1, int(attrs["level_patch_size"] / downsample))
        thumbnail_coords = np.zeros_like(coords)
        for i, (x, y, _, _) in enumerate(coords):
            x, y = get_x_y_to(
                (x, y),
                (level_width, level_height),
                (thumbnail_width, thumbnail_height),
                integer=True,
            )
            thumbnail_coords[i, :] = [x, y, thumbnail_patch_size, thumbnail_patch_size]

        # Pre Class HM:
        for i, target in enumerate(self.label_encoder.classes_):
            # Softmax the attention scores
            maxtw = softmax(tw)
            class_scores = maxtw * logits[:, i]
            heatmaps_class, background = self.fill_heatmap(
                size=(thumbnail_width, thumbnail_height),
                xywh=thumbnail_coords,
                scores=class_scores,
            )
            overlay = self.overlay_heatmap_on_thumbnail(
                thumbnail,
                heatmaps_class,
                background,
                smooth=smooth,
                alpha=alpha,
            )
            overlays[f"class:{target}"] = overlay

        # Attention HM:
        attn_scores = softmax(tw)
        heatmap_attn, background = self.fill_heatmap(
            size=(thumbnail_width, thumbnail_height),
            xywh=thumbnail_coords,
            scores=attn_scores,
        )
        overlay = self.overlay_heatmap_on_thumbnail(
            thumbnail,
            heatmap_attn,
            background,
            smooth=smooth,
            alpha=alpha,
        )
        overlays["attn"] = overlay

        if save:
            for i, key in enumerate(overlays.keys()):
                save_dir = os.path.join(save, key)
                safe_mkdir(save_dir)
                overlay = Image.fromarray(overlays[key].astype("uint8"), "RGB")
                overlay.save(os.path.join(save_dir, f"{ID}_{key}_heatmap.png"))
        return overlays

    def make_all(self, dst, downsample=16):
        table = self.table
        ID = table["ID"].values
        for n, o in enumerate(ID):
            self.make_images(ID=o, save=dst)
        return self

    def fill_heatmap(self, size, xywh, scores):
        """
        Creates a normalized heatmap from scores and an infomat matrix.

        Args:

        Returns:
        """
        heatmap = -1 * np.ones((size[1], size[0]), dtype=np.float32)
        for i, tile in enumerate(xywh):
            x, y = tile[0], tile[1]
            w, h = tile[2], tile[3]
            heatmap[y : y + h, x : x + w] = scores[i]
        background = heatmap == -1
        return heatmap * (background - 1) * -1, background

    def overlay_heatmap_on_thumbnail(
        self,
        thumbnail,
        heatmap,
        background,
        smooth: int | None = None,
        alpha: float = 0.5,
    ):
        """
        Overlays the normalized heatmap onto a higher magnification thumbnail of the whole-slide image and saves it.

        Args:
        Returns:
        """

        # Normalize the heatmap to [0, 1].
        heatmap_filtered = np.ma.masked_array(heatmap, mask=background)
        norm = Normalize(vmin=heatmap_filtered.min(), vmax=heatmap_filtered.max())
        heatmap_normalized = norm(heatmap_filtered)

        # gaussian smoothing
        if smooth is not None:
            heatmap_normalized = gaussian_filter(heatmap_normalized, sigma=smooth)

        # Apply perceptually nice colormap only on non-background pixels in the heatmap
        colormap = plt.get_cmap("jet")
        colored_heatmap = colormap(heatmap_normalized)
        colored_heatmap = (colored_heatmap[..., :3] * 255).astype(np.uint8)
        colored_heatmap[..., :3][background] = 0

        blend = blend_images(
            thumbnail, colored_heatmap, background_color=(0, 0, 0), alpha=alpha
        )

        return blend
