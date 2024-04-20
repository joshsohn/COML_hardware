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
                for reg_P in 2e-3
                    do
                    for reg_k_R in 2e-3
                    do
                        for k_R_scale in 2 3 4
                        do
                            echo "seed = $seed, M = $M, pnorm_init = $p, p_freq = $p_freq, meta_epochs = 1000, reg_P = $reg_P reg_k_R = $reg_k_R k_R_scale = $k_R_scale output_dir = reg_P_${reg_P}_reg_k_R_${reg_k_R}_k_R_scale_${k_R_scale}"
                            sbatch train_single.sh $seed $M --pnorm_init $p --p_freq $p_freq --meta_epochs 1000 --reg_P $reg_P --reg_k_R $reg_k_R --k_R_scale $k_R_scale --output_dir "reg_P_${reg_P}_reg_k_R_${reg_k_R}_k_R_scale_${k_R_scale}"
                        done
                    done
                done
            done
        done
    done
done