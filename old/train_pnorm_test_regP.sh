#!/bin/bash
#SBATCH --job-name train_pnorm_test_regP
#SBATCH -N 1
#SBATCH --exclusive

for seed in 0
do
    for M in 25
    do
        echo "seed = $seed, M = $M"
        for p in 2.0
        do
            for p_freq in 50
            do
                # The p-norm param would not change too much, sweeping with one p_freq should be fine
                for reg_P in 2e-3 3e-3 4e-3 5e-3 6e-3
                do
                    sbatch train_pnorm_single.sh $seed $M --pnorm_init $p --p_freq $p_freq --meta_epochs 4000 --reg_P $reg_P --output_dir 'diagnose_regP'
                done
            done
        done
    done
done