#!/bin/bash
#SBATCH --job-name train_reg_P_many
#SBATCH -N 1
#SBATCH --exclusive

for seed in 0
do
    for M in 50
    do
        for p in 2.0
        do
            for p_freq in 2000
            do
                for reg_P in 1 1e-1 5e-1
                do
                    echo "seed = $seed, M = $M, pnorm_init = $p, p_freq = $p_freq, meta_epochs = 1000, reg_P = $reg_P output_dir = z_up_reg_P_${reg_P}_kRz"
                    sbatch train_pnorm_single.sh $seed $M --pnorm_init $p --p_freq $p_freq --meta_epochs 1000 --reg_P $reg_P --output_dir "z_up_reg_P_${reg_P}_kRz"
                done
            done
        done
    done
done