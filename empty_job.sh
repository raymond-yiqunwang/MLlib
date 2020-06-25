#!/bin/bash
#SBATCH -A a9009                # Allocation
#SBATCH -p a9009                # Queue
#SBATCH -t 02:00:00             # Walltime/duration of the job
#SBATCH -N 1                    # Number of Nodes
#SBATCH --ntasks-per-node=24    # Number of Cores (Processors)
#SBATCH --mem=50G               # Memory per node in GB needed for a job.
#SBATCH --job-name="test"       # Name of job
##SBATCH --nodelist=qnode5056,qnode5057

sleep 1h
