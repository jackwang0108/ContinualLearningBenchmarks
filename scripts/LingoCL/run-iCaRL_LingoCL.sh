#!/bin/bash

root_dir=$(realpath "$(dirname "$(realpath "$0")")/../../")

for ((i = 1; i <= 10; i++)); do
    echo "Running iCaRL-LingoCL for the $i time, do not type in the name of this run, the script fills automatically!"
    echo "iCaRL-LingoCL-$i-th-Run" > runtime-iCaRL-LingoCL.txt
    cd "$root_dir" && python -m src.LingoCL.iCaRL_LingoCL < runtime-iCaRL-LingoCL.txt
done

rm runtime-iCaRL-LingoCL.txt
