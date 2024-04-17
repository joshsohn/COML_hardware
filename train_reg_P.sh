#!/bin/bash

for seed in 0
do
    for M in 50
    do
        for p in 2.0
        do
            for p_freq in 2000
            do
                for reg_P in 1e-1 5e-1 1
                do
                    echo "seed = $seed, M = $M, pnorm_init = $p, p_freq = $p_freq, meta_epochs = 1000, reg_P = $reg_P output_dir = hardware_z_up_$reg_P"
                    python3 train_z_up.py $seed $M --pnorm_init $p --p_freq $p_freq --meta_epochs 1000 --reg_P $reg_P --output_dir "hardware_z_up_$reg_P"
                done
            done
        done
    done
done