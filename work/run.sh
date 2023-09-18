#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=qed-fci
#SBATCH --mem=0

module purge
module load gcc/9.2.0
module load binutils/2.26
module load cmake-3.6.2 

export OMP_NUM_THREADS=28;
export MKL_NUM_THREADS=28;
export PYSCF_MAX_MEMORY=$SLURM_MEM_PER_NODE;
echo $PYSCF_MAX_MEMORY

source /home/yangjunjie/intel/oneapi/setvars.sh --force;
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

export TMPDIR=/scratch/global/yangjunjie/
export PYSCF_TMPDIR=/scratch/global/yangjunjie/

export PYSCF_TMPDIR=/scratch/global/yangjunjie/;
cd ../wick-main/; export PYTHONPATH=$PWD; cd -;
export PYTHONPATH=/home/yangjunjie/packages/pyscf/pyscf-main/:$PYTHONPATH;
export PYTHONUNBUFFERED=TRUE;

python gen-cceqs.py

