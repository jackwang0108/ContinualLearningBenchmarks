#!/bin/bash

root_dir=$(realpath "$(dirname "$(realpath "$0")")/../")

# check src/main.py for general args and src/LUCIR.py for algorithm args

cl_algo="LUCIR"

# running with paper hyper-parameters
cd "$root_dir" && python -m main --model "${cl_algo}" --backbone "resnet34" --dataset "cifar100" --num_tasks 10 --lr 0.1 --epochs 160 --batch_size 128 --each_buffer_size 20 --log_times 10 --gpu_id 0 --name "NAME OF THE LOGGING DIR" --message "LEAVE YOUR COMMENT OF THIS RUNNING" 


# a better hyper-parameters
cd "$root_dir" && python -m main --model "${cl_algo}" --backbone "resnet34" --dataset "cifar100" --num_tasks 10 --fixed_tasks --lr 0.1 --epochs 160 --batch_size 32 --each_buffer_size 20 --log_times 10 --gpu_id 0 --name "NAME OF THE LOGGING DIR" --message "LEAVE YOUR COMMENT OF THIS RUNNING" 

# use same buffer management policy and epochs for fair comparison
cd "$root_dir" && python -m main --model "${cl_algo}" --backbone "resnet34" --dataset "cifar100" --num_tasks 10 --fixed_tasks --lr 0.1 --epochs 100 --batch_size 32 --total_buffer_size 2000 --log_times 10 --gpu_id 0 --name "NAME OF THE LOGGING DIR" --message "LEAVE YOUR COMMENT OF THIS RUNNING" 