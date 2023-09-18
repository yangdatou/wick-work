#!/bin/bash
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --job-name=cceph
#SBATCH --output=/scratch/global/yangjunjie/slurm-%x-%j.log

module purge
module load PrgEnv-gnu/6.0.10
module load cmake/3.22.2
module load python

export MKLROOT="/opt/intel/compilers_and_libraries_2020.2.254/linux/mkl"
source $MKLROOT/bin/mklvars.sh intel64

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYSCF_MAX_MEMORY=$SLURM_MEM_PER_NODE

echo SLURM_NTASKS         = $SLURM_NTASKS
echo OMP_NUM_THREADS      = $OMP_NUM_THREADS
echo MKL_NUM_THREADS      = $MKL_NUM_THREADS
echo OPENBLAS_NUM_THREADS = $OPENBLAS_NUM_THREADS
echo PYSCF_MAX_MEMORY     = $PYSCF_MAX_MEMORY

source /home/yangjunjie/intel/oneapi/setvars.sh --force;
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH

export TMPDIR=/scratch/global/yangjunjie/$SLURM_JOB_NAME-$SLURM_JOB_ID/
export PYSCF_TMPDIR=TMPDIR; echo TMPDIR=$TMPDIR
mkdir -p $TMPDIR

export LOG_TMPDIR=$SLURM_SUBMIT_DIR/out/$SLURM_JOB_NAME-$SLURM_JOB_ID/
echo LOG_TMPDIR=$LOG_TMPDIR; mkdir -p $LOG_TMPDIR
ln -s /scratch/global/yangjunjie/slurm-$SLURM_JOB_NAME-$SLURM_JOB_ID.log $LOG_TMPDIR/slurm.out

cd ../wick-main/; export PYTHONPATH=$PWD; cd -;
export PYTHONPATH=/home/yangjunjie/packages/pyscf/pyscf-main/:$PYTHONPATH;
export PYTHONUNBUFFERED=TRUE;

time mpiexec -n $SLURM_NTASKS \
python gen-cceqs.py