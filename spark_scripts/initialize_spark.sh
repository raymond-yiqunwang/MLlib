#!/bin/bash

set -a
. /projects/a9009/rwang/spark_scripts/spark-env.sh 
set +a

# start master
$SPARK_HOME/sbin/spark-daemon.sh start org.apache.spark.deploy.master.Master 1 
sleep 5

# start slaves
NODE_LIST=$( scontrol show hostnames $SLURM_JOB_NODELIST )
for SLAVE_NODE in $NODE_LIST; do
  echo "-- Starting SPARK Slave at host $SLAVE_NODE"
  timeout 30 ssh $SLAVE_NODE \
   "set -a; \
    SPARK_LOG_DIR=$SPARK_LOG_DIR; \
    SPARK_WORKER_DIR=$SPARK_WORKER_DIR; \
    SPARK_WORKER_CORES=$SPARK_WORKER_CORES; \
    SPARK_WORKER_MEMORY=$SPARK_WORKER_MEMORY; \
    set +a; \
    $SPARK_HOME/sbin/start-slave.sh spark://$SPARK_MASTER_HOST:$SPARK_MASTER_PORT"
done

