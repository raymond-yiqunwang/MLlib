#!/bin/bash
#SBATCH -A a9009                # Allocation
#SBATCH -p a9009                # Queue
#SBATCH -t 00:10:00             # Walltime/duration of the job
#SBATCH -N 2                    # Number of Nodes
#SBATCH --ntasks-per-node=24    # Number of Cores (Processors)
#SBATCH --mem=50G               # Memory per node in GB needed for a job.
#SBATCH --job-name="test"       # Name of job
#SBATCH --nodelist=qnode5056,qnode5057

module load spark/2.3.0

. /projects/a9009/rwang/spark_scripts/initialize_spark.sh 

### output for the user
echo ""
echo ""
echo "########## Web UI Information below ##########"
echo ""
SPARK_MASTER_IP=`hostname --ip-address`
echo "Spark master host node name: " $SPARK_MASTER_HOST "IP: " $SPARK_MASTER_IP
echo ""
echo "You can view the master UI at: " $SPARK_MASTER_IP:$SPARK_MASTER_PORT
echo ""
echo "Job execution details are available at: " $SPARK_MASTER_IP:$SPARK_MASTER_WEBUI_PORT
echo ""
echo "########## Web UI Information above ##########"
echo ""
echo ""


spark-submit \
  --master spark://$SPARK_MASTER_HOST:7077 \
  compute_pi.py 1000

. /projects/a9009/rwang/spark_scripts/cleanup_spark.sh
