#!/bin/bash
#

# This is an example bash script that is used to submit a job
# to the cluster.
#
# Typcially, the # represents a comment.However, #SBATCH is
# interpreted by SLURM to give an option from above. As you 
# will see in the following lines, it is very useful to provide
# that information here, rather than the command line.

# Name of the job - You MUST use a unique name for the job
#SBATCH -J p6d15c10

# Standard out and Standard Error output files
# Each job should have a unique file name; otherwise, all of the
# output goes to one file and becomes hard to read/analyze.
#SBATCH -o p6d15c10.out
#SBATCH -e p6d15c10.err

# In order for this to send emails, you will need to remove the
# space between # and SBATCH for the following 2 commands.
# Specify the recipient of the email
# SBATCH --mail-user=abc1234@rit.edu

# Notify on state change: BEGIN, END, FAIL or ALL
# SBATCH --mail-type=ALL

# Multiple options can be used on the same line as shown below.
# Here, we set the partition and the number of cores to use.
#SBATCH -p kgcoe-mps -n 6

#
# Your job script goes below this line.
#

# If the job that you are submitting is not sequential,
# then you MUST provide this line...it tells the node(s)
# that you want to use this implementation of MPI. If you
# omit this line, your results will indicate failure.
module load openmpi-1.10-x86_64

# Place your srun command here
# Notice that you have to provide the number of processes that
# are needed. This number needs to match the number of cores
# indicated by the -n option. If these do not, your results will
# not be valid or you may have wasted resources that others could
# have used.
srun -n $SLURM_NPROCS mpi_hh -d 15 -c 10
