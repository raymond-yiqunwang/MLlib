#!/bin/bash

set -a;
. /projects/a9009/rwang/spark_scripts/spark-env.sh
set +a;

# stop slaves
NODE_LIST=$( scontrol show hostnames $SLURM_JOB_NODELIST )
for SLAVE_NODE in $NODE_LIST; do
    echo "-- Stopping SPARK Slave at host $SLAVE_NODE"
    timeout 30 ssh $SLAVE_NODE \
       "set -a; \
        SPARK_LOG_DIR=$SPARK_LOG_DIR; \
        set +a; \
        ${SPARK_HOME}/sbin/spark-daemon.sh stop org.apache.spark.deploy.worker.Worker 1"
done

# stop master
${SPARK_HOME}/sbin/spark-daemon.sh stop org.apache.spark.deploy.master.Master 1

