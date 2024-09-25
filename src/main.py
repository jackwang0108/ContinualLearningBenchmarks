# Standard Library
import argparse
import datetime
import importlib
from pathlib import Path
from typing import cast, Optional, Protocol

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# My Library
import src
import utils.datasets
from utils.datasets import CLDatasetGetter
from utils.helper import (get_logger, 
                          set_random_seed
                          )
from utils.helper import (plot_matrix, draw_image)
from utils.helper import (get_probas, get_pred, to_khot)
from utils.helper import (get_top1_acc,
                          get_backward_transfer
                          )
from utils.annotation import (
    TaskLearner,
    PerformanceFunc,
    CLAbilityTester
)


logger = None
device: torch.device = None
writer: SummaryWriter = None

class CLModule(Protocol):
    """
    CLModule defines the protocols (a set of functions) that a continual learning algorithm implementation (e.g. ./src/iCaRL.py) must has.
    """
    
    def get_args(self, argument_list: list[str]) -> tuple[argparse.Namespace, list[str]]:
        pass
    
    def get_model(self) -> nn.Module:
        pass

def train_epoch(model: nn.Module, train_loader: DataLoader):
    pass

@torch.no_grad()
def test_epoch(mode: nn.Module, test_loader: DataLoader):
    pass


def get_task_learner(args: argparse.Namespace) -> TaskLearner:
    
    num_task_learned = 0
    
    def task_learner(task_id: int, current_task: list[str], num_cls_per_task:int, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader) -> nn.Module:
        pass
    
    return task_learner


def continual_learning(main_args: argparse.Namespace):
    
    dataset_getter = CLDatasetGetter(
        dataset=main_args.dataset, task_num=main_args.num_tasks, fixed_task=main_args.fixed_tasks
    )



def get_args() -> tuple[CLModule, argparse.Namespace, argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    
    # basic arguments
    parser.add_argument("--model", type=str,
                        default="finetune", choices=src.avaliable_model, help="continual learning model to train")
    parser.add_argument("--dataset", type=str,
                        default="cifar100", choices=utils.datasets.avaliable_datasets, help="datasets to use")
    parser.add_argument("--num_tasks", type=int, default=10, help="number of tasks")
    parser.add_argument("--fixed_tasks", default=False, action="store_true", help="if use predefined task list")

    # logging arguments
    parser.add_argument("--name", type=str, default="", help="name of the current training")
    parser.add_argument("--message", type=str, default="", help="use the given message as the training message")
    
    # other arguments
    parser.add_argument("--seed", type=int, default=2024, help="value of the random seed")
    parser.add_argument("--gpu_id", type=str, default="0", help="which gpu to use")
    
    main_args, remaining_args = parser.parse_known_args()
    
    # logging
    prefix = "" if main_args.name == "" else "-"
    current_time = datetime.datetime.now().strftime("%m-%d %H.%M") 
    main_args.name = f"{main_args.name}{prefix}{current_time}"
    main_args.message += ("" if main_args.message == "" else ". ")  + f"Running on {current_time}"
    
    global logger, writer
    writer = SummaryWriter(log_dir := f"log/{main_args.model}/{main_args.name}")
    logger = get_logger(Path(log_dir) / "running.log")
    
    # load continual learning module
    model_module = cast(CLModule, importlib.import_module(f"src.{main_args.model}"))
    
    model_args, unknow_args = model_module.get_args(remaining_args)
    
    for args_name, args in {
        "main args": main_args, 
        "model args": model_args, 
        "unknown args": dict(zip(unknow_args[::2], unknow_args[1::2]))}.items():
        
        logger.success(f"{args_name}:")
        
        for key, value in (vars(args) if args_name != "unknown args" else args).items():
            logger.info(f"\t{key}: {value}")
    
    return model_module, main_args, model_args, unknow_args


def main():
    # get command line arguments
    model_module, main_args, model_args, unknow_args = get_args()
    
    # fix all random status
    set_random_seed(seed=main_args.seed)
    
    # set device
    global device
    device = torch.device(f"cuda:{main_args.gpu_id}" if main_args.gpu_id != "cpu" else "cpu")
    
    # TODO: 模型支持backbone
    # TODO: MNIST支持下载

if __name__ == "__main__":
    main()