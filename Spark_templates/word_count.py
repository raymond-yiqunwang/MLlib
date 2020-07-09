import findspark
SPARK_HOME = '/opt/spark-3.0.0-bin-hadoop2.7/'
findspark.init(SPARK_HOME)

import sys
import operator
from pyspark.sql import SparkSession

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python wordcount <filename>", file=sys.stderr)
        sys.exit(-1)

    spark = SparkSession\
        .builder\
        .appName("PythonWordCount")\
        .getOrCreate()

    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    counts = lines.flatMap(lambda x: x.split(' ')) \
                  .map(lambda x: (x, 1)) \
                  .reduceByKey(operator.add)
    
    output = counts.collect()
    for (word, count) in output:
        print("%s: %i" % (word, count))
    
    spark.stop()
