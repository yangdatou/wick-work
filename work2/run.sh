#!/bin/bash
#SBATCH --partition=serial
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=0
#SBATCH --job-name=cceph
#SBATCH --output=/scratch/global/yangjunjie/slurm-%x-%j.log

module purge
module load gcc/9.2.0
module load binutils/2.26
module load cmake-3.6.2 

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

# time python gen-cceqs.py 2 1 4 1 > $LOG_TMPDIR/2141.log
# time python gen-cceqs.py 2 2 4 1 > $LOG_TMPDIR/2241.log
# time python gen-cceqs.py 1 1 4 0 > $LOG_TMPDIR/1140.log
# time python gen-cceqs.py 1 2 4 0 > $LOG_TMPDIR/1240.log
# time python gen-cceqs.py 1 4 3 0 > $LOG_TMPDIR/1430.log
# time python gen-cceqs.py 1 6 3 0 > $LOG_TMPDIR/1630.log
# time python gen-cceqs.py 1 8 3 0 > $LOG_TMPDIR/1830.log

time python gen-cceqs.py 1 1 3 0 > $LOG_TMPDIR/1130.log
time python gen-cceqs.py 1 1 4 0 > $LOG_TMPDIR/1140.log
time python gen-cceqs.py 1 1 5 0 > $LOG_TMPDIR/1150.log

time python gen-cceqs.py 1 2 3 0 > $LOG_TMPDIR/1230.log
time python gen-cceqs.py 1 2 4 0 > $LOG_TMPDIR/1240.log
time python gen-cceqs.py 1 2 5 0 > $LOG_TMPDIR/1250.log

time python gen-cceqs.py 1 4 3 0 > $LOG_TMPDIR/1430.log
time python gen-cceqs.py 1 4 4 0 > $LOG_TMPDIR/1440.log
time python gen-cceqs.py 1 4 5 0 > $LOG_TMPDIR/1450.log
