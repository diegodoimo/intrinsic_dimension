#!/bin/bash
#SBATCH --account=Sis22_laio_0
#SBATCH --partition=m100_usr_prod
##SBATCH --qos=normal
#SBATCH --qos=m100_qos_dbg          	
#SBATCH --nodes=1
#SBATCH --time=0:59:00             		#format: HH:MM:SS
#SBATCH --ntasks-per-node=1        		#4 tasks uso una sola cpu (SMT)
#SBATCH --cpus-per-task=32        		#128cpu/nodo 64x2 socket
#SBATCH --gres=gpu:1            			#1 gpus per node out of 4
#SBATCH --mem=32000                		#memory per node out of 256000MB
#SBATCH --job-name=intrinsic_dim

#source module_load
module load profile/deeplrn
module load autoload cineca-ai/2.1.0
export OMP_NUM_THREADS=16

srun  -N1 --gres gpu:1 python3 simulation_poisson_process.py --nsamples $nsamples \
      --n $n_max --ndata $ndata --id $id --results_path = './results'
