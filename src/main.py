# Standard Library
import argparse
import datetime
import importlib
from pathlib import Path
from argparse import Namespace
from typing import cast, Optional, Protocol

# Third-Party Library
import numpy as np

# Torch Library
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# My Library
import src
import model
import utils.datasets
from utils.datasets import CLDatasetGetter
from utils.helper import get_logger, set_random_seed
from utils.helper import plot_matrix, draw_image
from utils.helper import get_top1_acc, get_backward_transfer
from utils.helper import (
    get_top1_acc,
    get_backward_transfer,
    get_last_setp_accuracy,
    get_average_incremental_accuracy,
    get_forgetting_rate,
    CLMetrics,
)
from utils.annotation import (
    Task,
    Images,
    Labels,
    TaskLearner,
    PerformanceFunc,
    CLAbilityTester,
)
from model.base import ContinualLearningModel


logger = None
device: torch.device = None
writer: SummaryWriter = None


class CLAlgoModule(Protocol):
    """
    CLModule defines the protocols (a set of functions) that a continual learning algorithm implementation (e.g. ./src/iCaRL.py) must has.
    """

    def get_args(self, argument_list: list[str]) -> tuple[Namespace, list[str]]:
        pass

    def get_model(self, backbone: nn.Module) -> nn.Module:
        pass

    def get_preds(self, cl_model: nn.Module) -> torch.FloatTensor:
        pass

    @torch.no_grad()
    def test_epoch(
        cl_model: nn.Module,
        test_loader: DataLoader,
        perf_func: PerformanceFunc,
        num_total_class: int,
    ):
        pass

    def prepare_continual_learning(self, module_args: Namespace, **kwargs) -> None:
        pass

    def prepare_new_task(self, module_args: Namespace, **kwargs):
        pass

    def finish_new_task(self, module_args: Namespace, **kwargs):
        pass

    def get_task_learner(
        self, main_args: Namespace, module_args: Namespace, **kwargs
    ) -> TaskLearner:
        pass


def get_continual_learning_ability_tester(
    main_args: Namespace, module_args: Namespace, cl_algo_module: CLAlgoModule
) -> CLAbilityTester:

    task_num = main_args.num_tasks
    cl_matrix = np.zeros((task_num, task_num))

    metrics_getter = CLMetrics(
        {
            "Backward Transfer": get_backward_transfer,
            "Forgetting Rate": get_forgetting_rate,
            "Last Step Accuracy": get_last_setp_accuracy,
            "Average Incremental Accuracy": get_average_incremental_accuracy,
        }
    )

    @torch.no_grad
    def continual_learning_ability_tester(
        task_id: int,
        current_task: Task,
        learned_tasks: list[Task],
        learned_task_test_loaders: list[DataLoader],
        cl_model: ContinualLearningModel,
    ) -> tuple[np.ndarray, dict[str, float]]:
        nonlocal cl_matrix, metrics_getter

        # test on all tasks, including previous and current task
        for i, previous_test_loader in enumerate(learned_task_test_loaders):

            cl_matrix[i, task_id] = cl_algo_module.test_epoch(
                cl_model,
                previous_test_loader,
                get_top1_acc,
                sum(len(lt) for lt in learned_tasks),
            )

            logger.info(
                f"\ttest on task {i}, test_acc={cl_matrix[i, task_id]: .2f}, {learned_tasks[i]}"
            )

        # calculate continual learning ability metrics and log to summarywriter
        if task_id >= 1:
            current_cl_matrix = cl_matrix[: task_id + 1, : task_id + 1]

            current_metrics = metrics_getter(current_cl_matrix, "mean")

            # log to summarywriter
            for metric_name, metric_value in current_metrics.items():
                if isinstance(metric_value, (float, int)):
                    writer.add_scalar(
                        tag=f"Continual Learning Metrics/{metric_name}",
                        scalar_value=metric_value,
                        global_step=task_id,
                    )

        # draw heatmap of cl_matrix and log to summarywriter
        writer.add_figure(
            tag=f"cl_matrix/{main_args.name}",
            figure=plot_matrix(cl_matrix, task_id),
            global_step=task_id,
        )

        return cl_matrix, current_metrics if task_id >= 1 else {}

    return continual_learning_ability_tester


def continual_learning(
    main_args: Namespace,
    module_args: Namespace,
    cl_model: ContinualLearningModel,
    cl_algo_module: CLAlgoModule,
):

    dataset_getter = CLDatasetGetter(
        dataset=main_args.dataset,
        task_num=main_args.num_tasks,
        fixed_task=main_args.fixed_tasks,
    )

    # logging
    logger.success(f"Model: {cl_model.__class__.__name__}")
    logger.success("Task List:")
    for task_id, task in enumerate(dataset_getter.tasks):
        logger.info(f"\tTask {task_id}: {task}")

    # get task learner and cl-ability tester
    task_learner: TaskLearner = cl_algo_module.get_task_learner(main_args, module_args)

    continual_learning_ability_tester: CLAbilityTester = (
        get_continual_learning_ability_tester(main_args, module_args, cl_algo_module)
    )

    test_dataset: Dataset
    train_dataset: Dataset
    learned_tasks: list[Task] = []
    learned_task_test_loaders: list[DataLoader] = []
    hparams_dict: dict[str, str | int | float] = {}

    # prepare continual learning
    cl_algo_module.prepare_continual_learning(module_args)

    for task_id, current_task, train_dataset, test_dataset in dataset_getter:

        # prepare for the new task
        cl_algo_module.prepare_new_task(
            module_args,
            cl_model=cl_model,
            task_id=task_id,
            current_task=current_task,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

        # get dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=main_args.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=dataset_getter.get_collate_fn(allow_default=True),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=main_args.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=dataset_getter.get_collate_fn(allow_default=True),
        )

        # learn the new task
        with cl_model.set_new_task(current_task):

            logger.success(f"{task_id=}, {current_task=}")

            training_watcher = {
                "Train Loss": torch.nan,
                "Test Top1 Acc": torch.nan,
            }

            # learn
            cl_model = task_learner(
                cl_model=cl_model,
                task_id=task_id,
                current_task=current_task,
                train_loader=train_loader,
                test_loader=test_loader,
                writer=writer,
                logger=logger,
                hparams_dict=hparams_dict,
                training_watcher=training_watcher,
            )

        # finish the new task
        cl_algo_module.finish_new_task(
            module_args,
            cl_model=cl_model,
            task_id=task_id,
            current_task=current_task,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            cls_id_mapper=dataset_getter.cls_id_mapper,
            logger=logger,
        )

        # save the test loader for continual learning testing
        learned_tasks.append(current_task)
        learned_task_test_loaders.append(test_loader)

        # test continual learning performance using standard metrics
        cl_matrix, metrics = continual_learning_ability_tester(
            task_id, current_task, learned_tasks, learned_task_test_loaders, cl_model
        )

        if metrics is not None:
            logger.debug(
                "\t"
                + ", ".join([f"{key}={value:.2f}" for key, value in metrics.items()])
            )

    # log hyper parameter
    logger.info("Hyper Parameters")
    for key, value in hparams_dict.items():
        logger.info(f"\t{key}: {value}")

    # log continual learning metrics
    logger.debug("Continual Learning Performance:")
    for key, value in metrics.items():
        logger.info(f"\t{key}: {value}")

    logger.info(f"Task learned: {dataset_getter.tasks}")
    logger.success("Finished Training")


def get_args() -> tuple[CLAlgoModule, Namespace, Namespace, list[str]]:
    parser = argparse.ArgumentParser()

    # basic arguments
    # fmt: off
    parser.add_argument("--model", type=str,
                        default="finetune", choices=src.avaliable_model, help="continual learning model to train")
    parser.add_argument("--backbone", type=str,
                        default="resnet18", choices=model.avaliable_backbone, help="vision backbones")
    parser.add_argument("--pretrained", default=False, action="store_true", help="if use pretrained backbone")
    parser.add_argument("--dataset", type=str,
                        default="cifar100", choices=utils.datasets.avaliable_datasets, help="datasets to use")
    parser.add_argument("--num_tasks", type=int, default=10, help="number of tasks")
    parser.add_argument("--fixed_tasks", default=False, action="store_true", help="if use predefined task list")
    
    # training arguments
    parser.add_argument("--lr", type=float, default=1e-3, help="starting learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="number of images in a batch")

    # logging arguments
    parser.add_argument("--name", type=str, default="", help="name of the current training")
    parser.add_argument("--message", type=str, default="", help="use the given message as the training message")
    parser.add_argument("--log_times", type=int, default=10, help="time of logging in a task")
    
    # other arguments
    parser.add_argument("--seed", type=int, default=2024, help="value of the random seed")
    parser.add_argument("--gpu_id", type=str, default="0", help="which gpu to use")
    # fmt: on

    main_args, remaining_args = parser.parse_known_args()

    # logging
    prefix = "" if main_args.name == "" else "-"
    current_time = datetime.datetime.now().strftime("%m-%d %H.%M.%S")
    main_args.name = f"{main_args.name}{prefix}{current_time}"
    main_args.message += (
        "" if main_args.message == "" else ". "
    ) + f"Running on {current_time}"

    global logger, writer
    writer = SummaryWriter(log_dir := f"log/{main_args.model}/{main_args.name}")
    logger = get_logger(Path(log_dir) / "running.log")

    # load continual learning module
    continual_learning_algorithm_module = cast(
        CLAlgoModule, importlib.import_module(f"src.{main_args.model}")
    )

    module_args, unknow_args = continual_learning_algorithm_module.get_args(
        remaining_args
    )

    for args_name, args in {
        "main args": main_args,
        "model args": module_args,
        "unknown args": dict(zip(unknow_args[::2], unknow_args[1::2])),
    }.items():

        logger.success(f"{args_name}:")

        for key, value in (vars(args) if args_name != "unknown args" else args).items():
            logger.info(f"\t{key}: {value}")

    return main_args, module_args, unknow_args, continual_learning_algorithm_module


def main():
    # get command line arguments
    continual_learning_algorithm_module: CLAlgoModule
    main_args, module_args, unknow_args, continual_learning_algorithm_module = (
        get_args()
    )

    # fix all random status
    set_random_seed(seed=main_args.seed)

    # set device
    global device
    device = torch.device(
        f"cuda:{main_args.gpu_id}" if main_args.gpu_id != "cpu" else "cpu"
    )

    # get the continual learning model
    backbone = model.get_backbone(main_args.backbone)
    continual_learning_model: ContinualLearningModel = (
        continual_learning_algorithm_module.get_model(backbone=backbone).to(device)
    )

    # continual learning
    continual_learning(
        main_args=main_args,
        module_args=module_args,
        cl_model=continual_learning_model,
        cl_algo_module=continual_learning_algorithm_module,
    )


if __name__ == "__main__":
    main()
