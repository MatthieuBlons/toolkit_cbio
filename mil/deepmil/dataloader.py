from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import torch
from slide.tile import EncodingSampler
from slide.utils import read_h5_coords, read_h5_features
from functools import reduce
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import os
from functools import partial


def collate_variable_size_old(
    batch,
):  # if constant_size is False we have to process a batch as a list of tensors (of different tilesxfeatures sizes)
    data = [item[0].unsqueeze(0) for item in batch]
    target = [torch.FloatTensor([item[1]]) for item in batch]
    return [data, target]


def collate_variable_size(batch, with_coords=False):
    """
    batch:
        if with_coords=True:
            [((feats, coords), target), ...]
        else:
            [(feats, target), ...]

    returns:
        if with_coords:
            (feats_list, coords_list), targets
        else:
            feats_list, targets
    """

    data = []
    targets = []

    if with_coords:
        for (feats, coords), target in batch:
            data.append((feats.unsqueeze(0), coords))
            targets.append(torch.as_tensor([target]).float())
    else:
        for feats, target in batch:
            data.append(feats.unsqueeze(0))
            targets.append(torch.as_tensor([target]).float())

    return [data, targets]


class WSIEncoded(Dataset):
    """
    Implements a dataset for already coded WSI.
    Each WSI is therefore a .npy array of size NxF with N the number of tiles
    of the WSI and F the number of features of the embeding space (usually 2048).
    Note: no transform method because this dataset is using numpy array as inputs.

    The target_table (labels of the different files) may have:
        * an ID column with the name of the images in it (without the extension)
        * a $args.target_name column, of course
        * a test columns, stating the test_fold number of each image.
    """

    def __init__(self, args, use_train, with_coords=False, predict=False):
        """Initialises the MIL model.

        Parameters
        ----------
        args : Namespace
            must contain :
                * target_path, str, path to the data info (.csv), with 'ID' column containing name of wsi.
                * wsi, str, path to the the output folder of a tile_image process. #embedded WSI (.h5) with name matching the 'ID' of target_path
                * target_name, str, name of the target variable (name of column in target_path)
                * device, torch.device
                * test_fold, int, number of the fold used as test.
                * feature_dim, int, number of dimension of the embedded space to keep. (0<x<2048)
                * n_tiles, int, if 0 : take all the tiles, will need custom collate_fn, else randomly picks $nb_tiles in each WSI.
                * train, bool, if True : extract the data s.t fold != test_fold, if False s.t. fold == testse_fold
                * sampler, str: tile sampler. dispo : random_sampler | random_biopsie
        """
        super(WSIEncoded, self).__init__()
        self.label_encoder = None
        self.args = args
        self.embeddings_dir = args.wsi_dir
        self.use_train = use_train
        self.with_coords = with_coords
        self.predict = predict
        self.target_table = pd.read_csv(args.target_path)
        (
            self.files,
            self.target_dict,
            self.stratif_dict,
            self.label_encoder,
        ) = self._make_db()
        self.constant_size = args.constant_size

    def __len__(self):
        return len(self.files)

    # Slide level encoding does not make sense in the MIL framework
    def __getitem__(self, idx):
        path = self.files[idx]
        target = self.target_dict[path]
        _, feats = self.get_embeddings(path)
        if self.args.wsi_enc == "tile":
            mat = feats[:, : self.args.feature_dim]
            mat, indices = self._select_tiles(path, mat)
            mat = torch.from_numpy(mat).float()
            if self.with_coords:
                _, coords = self.get_coords(path)
                coords = coords[indices, :2]
                coords = torch.from_numpy(coords).float()
                return (mat, coords), target
            else:
                return mat, target
        elif self.args.wsi_enc == "slide":
            mat = feats[: self.args.feature_dim]
            mat = torch.from_numpy(mat).float()
            return mat, target

    def _select_tiles(self, path, mat):
        """_select_tiles.
        Samples the tiles in the WSI.

        :param path: str: path of the current WSI.
        :param mat: ndarray: matrix of the embedded wsi.
        return ndarray: matrix of the subsampled WSI.
        """
        if self.use_train:
            train_sampler = EncodingSampler(
                sampler_name=self.args.sampler,
                feat_path=path,
                n_samples=self.args.n_tiles,
            )
            indices = train_sampler.sampler()
            mat = mat[indices, :]
        else:
            test_sampler = EncodingSampler(
                sampler_name=self.args.val_sampler,
                feat_path=path,
            )
            indices = test_sampler.sampler()
            mat = mat[indices, :]
        return mat, indices

    def get_embeddings(self, path):
        attrs, feats = read_h5_features(path)
        return attrs, feats

    def get_coords(self, path):
        attrs, coords = read_h5_coords(path)
        slide_size = attrs.get("level_size", None)
        if slide_size is not None:
            coords /= slide_size
        return attrs, coords

    def _make_db(self):
        """_make_db.
        Creates the dataset. Namely, populates the files list
        with the selected WSI.
        Populates also 3 dictionnary, with keys the elements of the files list
        and values :
            * target_dict : their target values
            * stratif_dict : their stratif values (present in the target_table).
            * sampler_dict : their associated TileSampler object.

        :return [files, target_dict, sampler_dict, stratif_dict, label_encoder]
        """
        table, label_encoder = self._transform_target()
        target_dict = dict()  # Key = path to the file, value=target
        stratif_dict = dict()
        names = table["ID"].values
        files_filtered = []
        for name in names:
            # do not check extension here?
            filepath = os.path.join(
                self.embeddings_dir, name + f".{self.args.wsi_format}"
            )
            if os.path.exists(filepath):
                if self._is_in_db(name):
                    files_filtered.append(filepath)
                    target_dict[filepath] = np.float32(
                        table[table["ID"] == name]["target"].values[0]
                    )
                    stratif_dict[filepath] = table[table["ID"] == name][
                        "stratif"
                    ].values[0]

        return files_filtered, target_dict, stratif_dict, label_encoder

    def _transform_target(self):
        """Adds to table a numerical encoding of the target.
        Each class is a natural number. Good format for classif using nn.CrossEntropy
        New columns is named "target"
        """
        tmp = self.target_table
        targets = tmp[self.args.target_name].values
        label_encoder = LabelEncoder().fit(targets)
        tmp["target"] = label_encoder.transform(targets)
        self.target_table = tmp
        return tmp, label_encoder

    def _is_in_db(self, name):
        """Do we keep the file in the dataset ?"""
        table = self.target_table
        is_in_db = True
        if "test" in table.columns and (not self.predict):
            is_in_train = (
                table[table["ID"] == name]["test"] != self.args.test_fold
            ).values[
                0
            ]  # "keep if i'm not test"
            is_in_test = (
                table[table["ID"] == name]["test"] == self.args.test_fold
            ).values[0]
            is_in_db = is_in_train if self.use_train else is_in_test
        return is_in_db


class Dataset_handler:
    """
    3 regimes are here possible :
        * We are training, we therefore want the train set split in train and val loaders.
        * We are not training, and not prediction (meaning testing): we need one loader for the test set.
        * We are not training, and predicting: we need one loader of the whole dataset.

    """

    def __init__(self, args, with_coords=False, predict=False):
        """
        Generates a validation dataset and a training dataset.
        If predict=True, the training dataset contains all the dataset.
        """
        self.args = args
        self.use_val = args.use_val
        self.num_class = args.num_class
        self.with_coords = with_coords
        self.predict = predict
        self.num_workers = args.num_workers
        self.dataset_train = self._get_dataset(use_train=True, with_coords=with_coords)
        self.dataset_test = self._get_dataset(use_train=False, with_coords=with_coords)
        self.train_sampler, self.val_sampler = self._get_sampler(
            self.dataset_train, use_val=args.use_val
        )

    def get_loader(self, training):
        """
        If training == False, therefore we are predictig : then taking the dataset_test
        that takes all the tiles, without a sampler (taking all the dataset)
        """
        if training:
            collate = (
                None
                if self.args.constant_size
                else partial(collate_variable_size, with_coords=self.with_coords)
            )
            dataloader_train = DataLoader(
                dataset=self.dataset_train,
                batch_size=self.args.batch_size,
                sampler=self.train_sampler,
                num_workers=self.num_workers,
                collate_fn=collate,
                drop_last=False,
            )
            dataloader_val = DataLoader(
                dataset=self.dataset_train,
                batch_size=1,
                sampler=self.val_sampler,
                num_workers=self.num_workers,
            )
            dataloaders = (dataloader_train, dataloader_val)
        else:  # Testing on the test set of predicting on the whole dataset (if predict = True)
            dataloaders = DataLoader(
                dataset=self.dataset_test,
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False,
            )
        return dataloaders

    def _get_dataset(self, use_train, with_coords):
        """_get_dataset.

        :param use_train: bool, if False, output dataset is composed of the
        testing fold, else of the training folds.
        :return EmbeddedWSI
        """
        dataset = WSIEncoded(
            self.args,
            use_train=use_train,
            with_coords=with_coords,
            predict=self.predict,
        )
        return dataset

    def _get_sampler(self, dataset, use_val=True):
        """_get_sampler.
        Samplers are iterators of the indices corresponding to the current training
        fold of the dataset. Training set is randomly divided in train and val. This
        is done through two different samplers, namely train_sampler and val_sampler.

        Integrates the strategic sampling.

        :param dataset: EmbeddedWSI corresponding to the current train fold.
        :param use_val: bool, if False, does not split the trainset in train/val.
        :return train_sampler, val_sampler
        """
        replacement = True
        if use_val:
            # to change
            # to change use labels_strat when there is no stratif (regression, cox...)
            # only use it with WeightedRandomSamplerFromList
            labels_strat = [dataset.stratif_dict[x] for x in dataset.files]
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=np.random.randint(100)
            )
            train_indices, val_indices = [
                x for x in splitter.split(X=labels_strat, y=labels_strat)
            ][0]
            labels_train_strat = np.array(labels_strat)[np.array(train_indices)]
            if self.args.no_strat_sampling:
                replacement = False
            train_sampler = WeightedRandomSamplerFromList(
                weights=self._get_weights_sampling(
                    labels_train_strat,
                    wr_whole_label=self.args.sample_wr_whole_label,
                    no_strat_sampling=self.args.no_strat_sampling,
                ),
                indices=train_indices,
                num_samples=len(train_indices),
                replacement=replacement,
            )
            val_sampler = SubsetRandomSampler(indices=val_indices)
        else:
            train_sampler = SubsetRandomSampler(list(range(len(dataset))))
            val_sampler = SubsetRandomSampler(list(range(len(dataset))))  # will be test
        return train_sampler, val_sampler

    def _get_weights_sampling(
        self, labels, wr_whole_label=False, no_strat_sampling=False
    ):
        """_get_weights_sampling.
        Computes the weights for sampling the batches.

        :param labels: labels w.r.t which attribute weights
        :param wr_whole_label: bool, if True simple oversampling to make
        each class equally probable. If False, equalize only the conditionnal expectation
        of each class w.r.t the target variable.

        For instance, let say the target variable is binary, T c {t1, t2}. B a confounder variable
        B c {b1, .., bn} with n values.
        if not wr_whole_label, we sample s.t P({T=ti} n {B=bj}) Vi,j.
        else: we sample s.t P({B=bi} | {T=t1}) = P({B=bi} | {T=t2}).

        """
        if no_strat_sampling:
            # print("no sampling strat will be used")
            weights = [1 for x in labels]

        elif wr_whole_label:
            # print("sampling conditioned by the numer of occurrences for each classes")
            cc = Counter(labels)
            weights = [1 / cc[x] for x in labels]

        else:
            # to keep?
            # print("sampling conditioned by the expectation of each class w.r.t the target variable")
            table = self.dataset_train.target_table
            target = self.args.target_name
            target_set = list(set(table[target].values))
            target_set = [str(x) for x in target_set]
            cc = Counter(labels)
            weights = []
            for l in labels:
                t_sample = l.split("_")[-2]
                t_op = target_set.index(t_sample)
                counts = 0
                for ind in range(len(target_set)):
                    v_op = l.split("_")  # si l = 'tnbc_lo_' je veux v_op = tnbc_hi_
                    v_op[-2] = target_set[ind]  # et en compter l'effectif.
                    v_op = reduce(lambda x, y: x + "_" + y, v_op)
                    counts += cc[v_op]
                weights.append(counts / cc[l])
        return weights


class WeightedRandomSamplerFromList(torch.utils.data.Sampler):
    """WeightedRandomSamplerFromList.
    Random sampler of a given list, each element being sampled
    with a probability proportional to its weight.
    """

    def __init__(self, weights, indices, num_samples, replacement=True, generator=None):
        """__init__.

        :param weights: list, list of weights. does not need to be normalized.
        :param indices: list, list of oject to sample from. must have the same lenght as weights.
        :param num_samples: int, how many elements are sampled at each iteration.
        :param replacement: bool. default=True.
        :param generator: random number generator. default=None.
        """
        assert len(weights) == len(indices)
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.indices = np.array(indices)

    def __iter__(self):
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=self.generator
        )
        return iter(list(self.indices[rand_tensor]))

    def __len__(self):
        return self.num_samples
