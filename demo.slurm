#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J bedlam_smpl                     # job name
#SBATCH --workdir=/home/cankeles/bedlam_smpl/BEDLAM # working directory
#SBATCH --gres=gpu:1  				# how many gpu you want.
#SBATCH --output=/home/cankeles/bedlam_smpl/BEDLAM/slurm/slurm-%j.out	# output file
#SBATCH --error=/home/cankeles/bedlam_smpl/BEDLAM/slurm/slurm-%j.err	# output error file
#SBATCH --time=15-00:00:00				# maximum running time.
#SBATCH --nodelist=romer1

########### CUDA Paths. You can change or delete them. #############
export PATH=/home/cankeles/cuda-11.0/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/cankeles/cuda-11.0/lib64
#########################################################


## Run command
python demo.py --cfg configs/demo_bedlam_cliff.yaml

echo "DONE!!!!"

