#!/bin/bash
#SBATCH --job-name train_pnorm_many_1
#SBATCH -N 1
#SBATCH --exclusive

for seed in 0
do
    for M in 25
    do
        echo "seed = $seed, M = $M"
        for p in 1.5 2.0 2.5 3.0
        do
            for p_freq in 20
            do
                for reg_P in 4e-3 8e-3
                do
                    sbatch train_pnorm_single.sh $seed $M --pnorm_init $p --p_freq $p_freq --meta_epochs 4000 --reg_P $reg_P --output_dir 'pnorm_varfreq_new'
                done
            done
        done
    done
done