#!/bin/bash

root_dir=$(realpath "$(dirname "$(realpath "$0")")/../")

for ((i = 1; i <= 10; i++)); do
    echo "Running iCaRL for the $i time, do not type in the name of this run, the script fills automatically!"
    echo "iCaRL-$i-th-Run" > runtime-iCaRL.txt
    cd "$root_dir" && python -m src.iCaRL < runtime-iCaRL.txt
done

rm runtime-iCaRL.txt
