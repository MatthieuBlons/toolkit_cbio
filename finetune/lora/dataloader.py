from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pandas as pd
import numpy as np
import os
import h5py
from PIL import Image
import albumentations as A
from sklearn.model_selection import ShuffleSplit
import torchvision.transforms as T
from mil.deepmil.utils import is_in_args


class HESignature(Dataset):
    """
    Implements a dataset.
    """

    def __init__(
        self,
        args,
        transform=None,
        use_train=True,
        predict=False,
    ):
        """
        Parameters
        ----------
        args : Namespace
            must contain :
                * img_dir, str, path to the he imgs directory.
                * input_path, str, path to the data info (.csv), with 'img_path' column containing name of input images.
                * target_path, str, path to the target .h5 file containing proteomic signatures
                * device, torch.device
                * test_fold, int, number of the fold used as test.
                * train, bool, if True : extract the data s.t fold != test_fold, if False s.t. fold == test_fold + Use data augmentation

        """
        super(HESignature, self).__init__()
        self.args = args
        self.ext = is_in_args(args, "extension", "jpeg")
        self.img_dir = args.img_dir
        self.use_train = use_train
        self.predict = predict
        self.transform = transform
        if not self.transform:
            self.transform = T.Compose([T.CenterCrop(args.img_size), T.ToTensor()])
        self.spatial_augmentations, self.color_augmentations = self.get_augmentations(
            training=use_train
        )
        self.input_table = pd.read_csv(args.input_path)
        self.img_to_fold = dict(
            zip(self.input_table["img_path"], self.input_table["test"])
        )
        self.target_path = args.target_path
        self.target_lables = self.fetch_labels() #typo
        self.input_files, self.target_dict = self._make_db()

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        path = self.input_files[idx]
        he = self.get_image(path)
        he = Image.fromarray(he).convert("RGB")
        he = self.transform(he)
        target = self.target_dict[path]
        return he, target

    def get_image(self, path):
        image = np.asarray(Image.open(path))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        if image.dtype not in [np.uint8, np.float32]:
            image = np.float32(image)
        if self.spatial_augmentations:
            image = self.spatial_augmentations(image=image)["image"]
        if self.color_augmentations:
            image = self.color_augmentations(image=image)["image"]
            image = np.clip(image, 0, 255)
        return image

    def get_signature(self, idx):
        path = self.input_files[idx]
        signature = self.target_dict[path]
        return signature

    def get_signature_from_path(self, path):
        name = os.path.basename(path)
        with h5py.File(self.target_path, "r") as f:
            img_paths = f["img_path"][:]
            img_to_idx = {p.decode("utf-8"): i for i, p in enumerate(img_paths)}
            try:
                idx = img_to_idx[name]
                return f["signature"][idx]
            except ValueError:
                print("Image path not found in target file .h5")

    def fetch_labels(self):
        with h5py.File(self.target_path, "r") as f:
            attrs = dict(f["signature"].attrs)
            features = attrs["feat"]
        return features

    def get_augmentations(
        self,
        training=True,
    ):
        if training:
            spatial_augmentations = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ]
            )
            # ask Guillaume about HedColorAugmentor
            color_augmentations = A.Compose(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                    A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(0.1, 1.5), p=0.1),
                    A.GaussNoise(var_limit=(0.05, 0.1), p=0.1),
                ]
            )
        else:
            spatial_augmentations = None
            color_augmentations = None

        return spatial_augmentations, color_augmentations

    def _make_db(self):
        """_make_db.
        Creates the dataset. Namely, populates the files list
        with the selected input images.
        Populates also dictionnaries, with keys the elements of the files list
        and values :
            * target_dict : their target signature values
        :return [files, target_dict]
        """
        target_dict = dict()  # Key = path to the file, value = target
        extension = ".jpeg"
        all_available = {f for f in os.listdir(self.img_dir) if f.endswith((extension))}
        with h5py.File(self.target_path, "r") as f:
            img_paths = f["img_path"][:]
            signatures = f["signature"][:]
            img_to_idx = {p.decode("utf-8"): i for i, p in enumerate(img_paths)}

        basenames = self.input_table["img_path"].values
        files_filtered = []
        # add tqdm
        for name in basenames:
            filepath = os.path.join(self.img_dir, name)
            if name not in all_available:
                continue
            if not self._is_in_db(name):
                continue
            try:
                idx = img_to_idx[name]
            except ValueError:
                print("Image path not found in target file .h5")
            target_dict[filepath] = signatures[idx]
            files_filtered.append(filepath)
        return (
            files_filtered,
            target_dict,
        )

    def _is_in_db(self, name):
        """Do we keep the file in the dataset ?"""
        fold = self.img_to_fold.get(name)
        if "test" in self.input_table.columns and (not self.predict):
            return (
                (fold != self.args.test_fold)
                if self.use_train
                else (fold == self.args.test_fold)
            )
        else:
            return True


class HEImage(Dataset):
    """
    Implements a dataset.
    """

    def __init__(
        self,
        args,
        transform=None,
        use_train=True,
        predict=False,
    ):
        """
        Parameters
        ----------
        args : Namespace
            must contain :
                * img_dir, str, path to the he imgs directory.
                * input_path, str, path to the data info (.csv), with 'img_path' column containing name of input images.
                * target_path, str, path to the target .h5 file containing proteomic signatures
                * device, torch.device
                * test_fold, int, number of the fold used as test.
                * train, bool, if True : extract the data s.t fold != test_fold, if False s.t. fold == test_fold + Use data augmentation

        """
        super(HEImage, self).__init__()
        self.args = args
        self.ext = is_in_args(args, "extension", "jpeg")
        self.img_dir = args.img_dir
        self.use_train = use_train
        self.predict = predict
        self.transform = transform
        if not self.transform:
            self.transform = T.Compose([T.CenterCrop(args.img_size), T.ToTensor()])
        self.spatial_augmentations, self.color_augmentations = self.get_augmentations(
            training=use_train
        )
        self.input_table = None
        if args.input_path is not None:
            self.input_table = pd.read_csv(args.input_path)
            self.img_to_fold = dict(
                zip(self.input_table["img_path"], self.input_table["test"])
            )
        self.input_files = self._make_db()

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        path = self.input_files[idx]
        img = self.get_image(path)
        img = Image.fromarray(img).convert("RGB")
        img = self.transform(img)
        return img

    def get_image(self, path):
        img = np.asarray(Image.open(path))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        if img.dtype not in [np.uint8, np.float32]:
            img = np.float32(img)
        if self.spatial_augmentations:
            img = self.spatial_augmentations(image=img)["image"]
        if self.color_augmentations:
            img = self.color_augmentations(image=img)["image"]
            img = np.clip(img, 0, 255)
        return img
    # work on numpy
    def get_augmentations(
        self,
        training=True,
    ):
        if training:
            spatial_augmentations = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ]
            )
            # ask Guillaume about HedColorAugmentor
            color_augmentations = A.Compose(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                    A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(0.1, 1.5), p=0.1),
                    A.GaussNoise(std_range=(0.05, 0.1), p=0.1),
                ]
            )
        else:
            spatial_augmentations = None
            color_augmentations = None

        return spatial_augmentations, color_augmentations

    def _make_db(self):
        """_make_db.
        Creates the dataset. Namely, populates the files list
        with the selected input images.
        Populates also dictionnaries, with keys the elements of the files list
        and values :
            * target_dict : their target signature values

        :return [files, target_dict]
        """
        files_filtered = []
        all_available = {f for f in os.listdir(self.img_dir) if f.endswith((self.ext))}
        if self.input_table is not None:
            basenames = self.input_table["img_path"].values
            # add tqdm
            for name in basenames:
                filepath = os.path.join(self.img_dir, name)
                if name not in all_available:
                    continue
                if not self._is_in_db(name):
                    continue
                files_filtered.append(filepath)
            return files_filtered
        else:
            for name in [*all_available]:
                filepath = os.path.join(self.img_dir, name)
                files_filtered.append(filepath)
            return files_filtered, None

    def _is_in_db(self, name):
        """Do we keep the file in the dataset ?"""
        fold = self.img_to_fold.get(name)
        if "test" in self.input_table.columns and (not self.predict):
            return (
                (fold != self.args.test_fold)
                if self.use_train
                else (fold == self.args.test_fold)
            )
        else:
            return True


class Dataset_handler:
    """
    3 regimes are here possible :
        * We are training, we therefore want the train set split in train and val loaders.
        * We are not training, and not prediction (meaning testing): we need one loader for the test set.
        * We are not training, and predicting: we need one loader of the whole dataset.

    """

    def __init__(self, args, img_only=False, preprocess=None, predict=False):
        """
        Generates a validation dataset and a training dataset.
        If predict=True, the training dataset contains all the dataset.
        """
        self.args = args
        self.preprocess = preprocess
        self.use_val = args.use_val
        self.predict = predict
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.dataset_train = self._get_dataset(img_only, use_train=True)
        self.dataset_test = self._get_dataset(img_only, use_train=False)
        self.train_sampler, self.val_sampler = self._get_sampler(
            self.dataset_train, use_val=args.use_val
        )

    def _get_dataset(self, img_only, use_train):
        """_get_dataset.

        :param use_train: bool, if False, output dataset is composed of the
        testing fold, else of the training folds.
        :return HESignature
        """
        if img_only:
            dataset = HEImage(
                self.args,
                transform=self.preprocess,
                use_train=use_train,
                predict=self.predict,
            )
        else:
            dataset = HESignature(
                self.args,
                transform=self.preprocess,
                use_train=use_train,
                predict=self.predict,
            )
        return dataset

    def get_train_dataset(self):
        return self.dataset_train

    def get_test_dataset(self):
        return self.dataset_test

    def get_loader(self, training):
        """
        If training == False, therefore we are predictig : then taking the dataset_test
        that takes all the tiles, without a sampler (taking all the dataset)
        """
        if training:
            dataloader_train = DataLoader(
                dataset=self.dataset_train,
                batch_size=self.batch_size,
                sampler=self.train_sampler,
                num_workers=self.num_workers,
                drop_last=True,
            )
            dataloader_val = DataLoader(
                dataset=self.dataset_train,
                batch_size=self.batch_size,
                sampler=self.val_sampler,
                num_workers=self.num_workers,
            )
            dataloaders = (dataloader_train, dataloader_val)
        else:  # Either testing on test fold or predicting on the whole dataset (if predict = True)
            dataloaders = DataLoader(
                dataset=self.dataset_test,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
        return dataloaders

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
        if use_val:
            # validation is done on 1/5th of the training dataset
            splitter = ShuffleSplit(
                n_splits=1, test_size=0.2, random_state=np.random.randint(100)
            )

            train_indices, val_indices = [
                x for x in splitter.split(X=dataset.input_files, y=dataset.input_files)
            ][0]

            train_sampler = SubsetRandomSampler(indices=train_indices)
            val_sampler = SubsetRandomSampler(indices=val_indices)

        else:
            train_sampler = SubsetRandomSampler(list(range(len(dataset))))
            val_sampler = SubsetRandomSampler(list(range(len(dataset))))

        return train_sampler, val_sampler
