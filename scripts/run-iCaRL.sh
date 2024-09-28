#!/bin/bash

root_dir=$(realpath "$(dirname "$(realpath "$0")")/../")

# check src/main.py for general args and src/iCaRL.py for algorithm args


# running with paper hyper-parameters
cd "$root_dir" && python -m main --model "iCaRL" --backbone "resnet34" --dataset "cifar100" --num_tasks 10 --lr 2 --epochs 70 --batch_size 128 --log_times 10 --gpu_id 0 --name "NAME OF THE LOGGING DIR" --message "LEAVE YOUR COMMENT OF THIS RUNNING" 


# a better hyper-parameters
cd "$root_dir" && python -m main --model "iCaRL" --backbone "resnet34" --dataset "cifar100" --num_tasks 10 --fixed_tasks --lr 2 --epochs 100 --batch_size 32 --log_times 10 --gpu_id 0 --name "NAME OF THE LOGGING DIR" --message "LEAVE YOUR COMMENT OF THIS RUNNING" 

# use randomly generated task lists
cd "$root_dir" && python -m main --model "iCaRL" --backbone "resnet34" --dataset "cifar100" --num_tasks 10 --lr 2 --epochs 70 --batch_size 32 --log_times 10 --gpu_id 0 --name "NAME OF THE LOGGING DIR" --message "LEAVE YOUR COMMENT OF THIS RUNNING" 