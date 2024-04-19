#!/bin/bash
#SBATCH --job-name train_many
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
                for reg_P in 1
                    do
                    for reg_k_R in 1e-5 1e-4 1e-2 1e-1
                    do
                        echo "seed = $seed, M = $M, pnorm_init = $p, p_freq = $p_freq, meta_epochs = 1000, reg_P = $reg_P reg_k_R=$reg_k_R output_dir = reg_P_${reg_P}_reg_k_R_${reg_k_R}"
                        sbatch train_single.sh $seed $M --pnorm_init $p --p_freq $p_freq --meta_epochs 1000 --reg_P $reg_P --reg_k_R $reg_k_R --output_dir "reg_P_${reg_P}_reg_k_R_${reg_k_R}"
                    done
                done
            done
        done
    done
done