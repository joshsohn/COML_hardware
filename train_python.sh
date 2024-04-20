#!/bin/bash

for seed in 0
do
    for M in 50
    do
        for p in 2.0
        do
            for p_freq in 2000
            do
                for reg_P in 2e-3
                    do
                    for reg_k_R in 2e-3
                    do
                        echo "seed = $seed, M = $M, pnorm_init = $p, p_freq = $p_freq, meta_epochs = 1000, reg_P = $reg_P reg_k_R=$reg_k_R output_dir = reg_P_${reg_P}_Kr_scale_${reg_k_R}"
                        python3 train_z_up_kR.py $seed $M --pnorm_init $p --p_freq $p_freq --meta_epochs 1000 --reg_P $reg_P --reg_k_R $reg_k_R --output_dir "reg_P_${reg_P}_Kr_scale_${reg_k_R}"
                    done
                done
            done
        done
    done
done