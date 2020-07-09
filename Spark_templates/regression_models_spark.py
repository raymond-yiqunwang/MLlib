#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import findspark
SPARK_HOME = '/opt/spark-3.0.0-bin-hadoop2.7/'
findspark.init(SPARK_HOME)

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("LinearRegressionWithElasticNet")\
        .getOrCreate()

    # TODO add timer
    # TODO check resource usage

    # Load training data
    data = spark.read.options(header=True, inferSchema=True)\
                .csv("../UCI_repo/Abalone/abalone.csv")
    data.cache()
    print(data.printSchema())
    print(data.describe().toPandas().transpose())

    indexer = StringIndexer(inputCol='Sex',outputCol='Sex_cat')
    indexed = indexer.fit(data).transform(data)

    for item in indexed.head(3):
        print(item)
        print('\n')

    assembler = VectorAssembler(inputCols=['Length', 'Diameter', 'Height', 'Whole weight',\
                                         'Shucked weight', 'Viscera weight', 'Shell weight',\
                                         'Sex_cat'], outputCol='features')

    output = assembler.transform(indexed)
    print(output.select('features', 'Rings').show(3))

    final_data = output.select('features', 'Rings')
    train_data, test_data = final_data.randomSplit([0.7, 0.3])
    test_data.describe().show(3)


    algo = 'linear_regression'
    #algo = 'decision_tree'

    if algo == 'linear_regression':
        from pyspark.ml.regression import LinearRegression
        regressor = LinearRegression(featuresCol = 'features', labelCol='Rings',
                                     maxIter=10, regParam=0.3, elasticNetParam=0.8)
    elif algo == 'decision_tree':
        from pyspark.ml.regression import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(featuresCol ='features', labelCol = 'Rings')

    lr_model = regressor.fit(train_data)

    # Print the coefficients and intercept for linear regression
    if algo == 'linear_regression':
        print("Coefficients: %s" % str(lr_model.coefficients))
        print("Intercept: %s" % str(lr_model.intercept))

        # Summarize the model over the training set and print out some metrics
        trainingSummary = lr_model.summary
        print("numIterations: %d" % trainingSummary.totalIterations)
        print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
        trainingSummary.residuals.show()
        print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
        print("r2: %f" % trainingSummary.r2)

    # test
    lr_predictions = lr_model.transform(test_data)
    print(lr_predictions.select("prediction", "Rings", "features").show(5))
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                                       labelCol="Rings", metricName="r2")
    print("R Squared (R2) on test data = %g" %lr_evaluator.evaluate(lr_predictions))

    test_result = lr_model.evaluate(test_data)
    print("Root Mean Squared Error (RMSE) on test data = %g" %test_result.rootMeanSquaredError)

    spark.stop()
