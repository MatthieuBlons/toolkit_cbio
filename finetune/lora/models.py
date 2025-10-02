"""
implementing models. DeepMIL implements a models that classify a whole slide image
"""

from torch.nn import MSELoss
from torch.optim import Adam
from mil.deepmil.utils import is_in_args
import torch
import torch
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
from finetune.lora.networks import LoRAFactory
from finetune.lora.dataloader import Dataset_handler
import numpy as np
from peft import LoraConfig
from torch.nn.utils import clip_grad_norm_
from torch import autocast, GradScaler


class Model(ABC):
    def __init__(self, args):
        self.args = args
        self.optimizers = []
        self.losses = {"global": []}
        self.metric = 0
        self.criterion = lambda: 1
        self.counter = {"epoch": 0, "batch": 0}
        self.network = torch.nn.Module()
        self.early_stopping = EarlyStopping(args=args)
        self.device = args.device
        self.ref_metric = args.ref_metric
        self.best_metrics = None
        self.best_ref_metric = None
        self.dataset = None
        self.writer = None

    @abstractmethod
    def optimize_parameters(self, input_batch, target_batch):
        pass

    @abstractmethod
    def make_state(self):
        pass

    @abstractmethod
    def predict(self, x):
        """Makes a prediction about the label of x.
        Prediction should be in numpy format.

        Parameters
        ----------
        x : torch.Tensor
            input
        """
        pass

    @abstractmethod
    def evaluate(self, x, y):
        pass

    def get_summary_writer(self):
        if "EVENTS_TF_FOLDER" in os.environ:
            directory = os.environ["EVENTS_TF_FOLDER"]
        else:
            directory = None
        self.writer = SummaryWriter(directory)

    def update_learning_rate(self, metric):
        for sch in self.schedulers:
            sch.step(
                metric
            )  # The epoch parameter in `scheduler.step()` was not necessary and is being deprecated

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def set_zero_grad(self):
        optimizers = self.optimizers
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            if optimizer is not None:
                optimizer.zero_grad()


class EarlyStopping:
    """Early stopping AND saver !"""

    def __init__(self, args):
        self.patience = args.patience
        self.counter = 0
        self.loss_min = None
        self.early_stop = False
        self.is_best = False
        self.filename = "model.pt.tar"

    def __call__(self, loss, state):
        if self.loss_min is None:
            self.loss_min = loss
            self.is_best = True
        elif self.loss_min <= loss:
            self.counter += 1
            self.is_best = False
            if not self.counter < self.patience:
                self.early_stop = True
        else:
            self.is_best = True
            self.loss_min = loss
            self.counter = 0
        self.save_checkpoint(state)

    def save_checkpoint(self, state):
        torch.save(state, self.filename)
        if self.is_best:
            shutil.copyfile(
                self.filename, self.filename.replace(".pt.tar", "_best.pt.tar")
            )


class HESingIF(Model):
    """
    Class implementing a LoRA framwork, for predicting proteo signatures from HE images.
    """

    def __init__(
        self,
        args,
        with_data=False,
    ):
        """
        args contains all the info for initializing the deepmil and its dataloader.

        :param args: Namespace. outputs of .arguments.get_arguments.
        :param with_data: bool, when True : set the data_loaders according to args. When False,

        The LoRA model is loaded without the data.
        """
        super(HESingIF, self).__init__(args)
        self.results_val = {"y_true": [], "pred": []}
        self.scores_dpp = []
        self.mean_train_loss = 0
        self.mean_val_loss = 0
        self.encoder = args.encoder
        self.lora_rank = is_in_args(args, "lora_r", 8)
        self.lora_alpha = is_in_args(args, "lora_a", 1)
        self.network, self.trainable, self.precision = self._get_network(
            rank=self.lora_rank, alpha=self.lora_alpha
        )
        optimizer = self._get_optimizer(args)
        self.optimizers = [optimizer]
        self.schedulers = self._get_schedulers(args)
        self.train_loader, self.val_loader = self._get_data_loaders(args, with_data)
        self.criterion = self._get_criterion(args.criterion)
        self.bayes = False

    def _get_network(self, rank=8, alpha=1):
        """_get_network.
        Initialize the network and transfer it on the cuda device.

        :return nn.Module: MIL network.
        """
        peft_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["qkv"],
            lora_dropout=0.05,
        )

        network = LoRAFactory(self.args, peft_cfg)
        network = network.to(self.args.device)
        trainable = [p for p in network.parameters() if p.requires_grad]
        precision = network.precision
        return network, trainable, precision

    def _get_optimizer(self, args):
        """_get_optimizer.

        :param args: Namespace. Outputs of .arguments.get_arguments.
        """
        if args.optimizer == "adam":
            optimizer = Adam(self.trainable, lr=args.lr, weight_decay=1e-5, eps=1e-7)
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.trainable, args.lr, momentum=0.9, weight_decay=1e-5, eps=1e-7
            )
        return optimizer

    def _get_schedulers(self, args):
        """_get_schedulers.
        Must be called after having define the optimizers (self._get_optimizer())

        Get the learning rate scheduler for the optimizer.
        :param args: Namespace. Outputs of .arguments.get_arguments.
        """
        if args.lr_scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
            schedulers = [
                scheduler(optimizer=o, patience=self.args.patience_lr, factor=0.3)
                for o in self.optimizers
            ]
        if args.lr_scheduler == "cos":
            # Use T_0 = 1 or allow first epochs to have a stable lr
            # add eta_min = minimal lr?
            schedulers = [
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer=o, T_0=1, T_mult=2
                )
                for o in self.optimizers
            ]
        return schedulers

    def _get_data_loaders(self, args, with_data):
        """_get_data_loaders.

        Gets the loaders for training.

        :param args: Namespace. Outputs of .arguments.get_arguments.
        :param with_data: bool, if TRUE, loads data, else not.
        """
        (
            train_loader,
            val_loader,
        ) = (
            None,
            None,
        )
        if with_data:
            data = Dataset_handler(args, preprocess=self.network.transform)
            train_loader, val_loader = data.get_loader(training=True)
        return train_loader, val_loader

    def _get_criterion(self, criterion):
        """_get_criterion.
        Constructor of the loss function

        :param criterion: str, loss_function
        """
        if criterion == "mse":
            criterion = MSELoss().to(self.args.device)
        return criterion

    def forward(self, x):
        out = self.network(x)
        return out

    def optimize_parameters(self, input_batch, target_batch):
        """optimize_parameters.

        Feed the network with a batch and optimize the parameter.
        Dataloader iterates (X, y), X being the data, y the label
        :param input_batch: data, X
        :param target_batch: label of X, y
        """
        scaler = GradScaler()

        self.set_zero_grad()

        input_batch = input_batch.to(self.args.device, self.precision["backbone"])

        target_batch = target_batch.to(self.args.device, torch.float32)

        with autocast(
            device_type=torch.device(self.device).type,
            enabled=(self.precision["backbone"] != torch.float32),
        ):
            output = self.forward(input_batch)
            loss = self.criterion(output, target_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(self.optimizers[0])
        clip_grad_norm_(self.trainable, 1.0)
        scaler.step(self.optimizers[0])
        scaler.update()

        return loss.detach().cpu().item()

    def _forward_no_grad(self, x):
        """_forward_no_grad.

        :param x: toch.tensor.
        """
        with torch.no_grad():
            with autocast(
                device_type=torch.device(self.device).type,
                enabled=(self.precision["backbone"] != torch.float32),
            ):
                # might run into dtype conflicts
                out = self.network(x)
        out = out.detach()
        return out

    def predict(self, x):
        """predict.

        :param x: torch.tensor image of shape wxhxc,
        :return proteo signature pred
        """
        x = x.to(self.args.device, self.precision["backbone"])
        pred = self._forward_no_grad(x)
        pred = pred.to("cpu", dtype=torch.float32)
        return pred.numpy()

    def evaluate(self, x, y):
        """
        takes x, y torch.Tensors.
        Predicts on x, stores y and the loss, and the outputs of the network.
        """
        x = x.to(self.args.device, self.precision["backbone"])
        y = y.to("cpu", dtype=torch.float32)
        pred = self._forward_no_grad(x)
        pred = pred.to("cpu", dtype=torch.float32)
        loss = self.criterion(pred, y)
        self.results_val["y_true"] += list(y.numpy())
        self.results_val["pred"] += list(pred.numpy())
        return loss.detach().cpu().item()

    def _compute_metrics(self, y_true, pred):
        metrics_dict = {}
        metrics_dict["epoch"] = self.counter["epoch"]
        metrics_dict["lr"] = [scheduler._last_lr[0] for scheduler in self.schedulers][0]
        # add validation meritc here
        return metrics_dict

    def _keep_best_metrics(self, metrics):
        """_keep_best_metrics.

        Stores the val metrics if this iteration is the best one, according to
        the ref_metric.
        :param metrics: dict of the validation metrics of the current epoch.
        """
        factor = self.args.sgn_metric
        if self.best_ref_metric is None:
            self.best_ref_metric = metrics[self.ref_metric]
            self.best_metrics = metrics
        if self.best_ref_metric * factor > metrics[self.ref_metric] * factor:
            self.best_ref_metric = metrics[self.ref_metric]
            self.best_metrics = metrics

    def flush_val_metrics(self):
        """flush_val_metrics.

        Once the forward pass for validation ends, computes the metrics, stores
        the best ones according to the ref metric.

        Metrics computed : lr, epochs, mean_train_loss,
        mean_val_loss.

        :return dict, key (name of the metric), value (value of the metric).
        """
        val_metrics = {}
        val_pred = np.array(self.results_val["pred"])
        val_y = np.array(self.results_val["y_true"])
        val_metrics = self._compute_metrics(val_y, val_pred)
        val_metrics["mean_train_loss"] = self.mean_train_loss
        val_metrics["mean_val_loss"] = self.mean_val_loss
        self._keep_best_metrics(val_metrics)

        # Re Initialize val_results for next validation
        self.results_val["y_true"] = []
        self.results_val["pred"] = []
        return val_metrics

    def make_state(self):
        """make_state.
        Creates a dictionnary checkpoint of the model.
        """
        dictio = {
            "state_dict": self.network.state_dict(),
            "state_dict_optimizer": self.optimizers[0].state_dict,
            "state_scheduler": self.schedulers[0].state_dict(),
            "inner_counter": self.counter,
            "args": self.args,
            "input_table": self.train_loader.dataset.input_table,
            "best_metrics": self.best_metrics,
        }
        return dictio


def load_model_from_path(model_path, device):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    args.device = device
    model = HESingIF(args)
    model.network.load_state_dict(checkpoint["state_dict"])
    return model
