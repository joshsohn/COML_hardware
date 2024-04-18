#!/bin/bash
#SBATCH --job-name train_fixed_pnorm
#SBATCH -N 1
#SBATCH --exclusive

for seed in 0
do
    for M in 25
    do
        echo "seed = $seed, M = $M"
        # for p in 1.5 1.7 2.0 2.3 2.5 2.7 3.0 3.3
        for p in 1.6 1.8 2.2 2.4 2.6 2.8 3.2 3.4
        do
            # fix p p_freq > meta_epochs
            for p_freq in 10000
            do
                for reg_P in 4e-3
                do
                    sbatch train_pnorm_single.sh $seed $M --pnorm_init $p --p_freq $p_freq --meta_epochs 4000 --reg_P $reg_P --output_dir 'pnorm_fixed_reg4'
                done
            done
        done
    done
done