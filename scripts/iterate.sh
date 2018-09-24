#!/bin/bash

# Iterate over all possible input arguments
for i in $(seq 0 17);
do
    python ./scripts/stabilityRun.py $i
done
echo $i
unset i

# python ./scripts/stabilityRun.py 2