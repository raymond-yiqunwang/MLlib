from __future__ import print_function
import findspark
SPARK_HOME = '/opt/spark-3.0.0-bin-hadoop2.7/'
findspark.init(SPARK_HOME)

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("LinearRegression")\
        .getOrCreate()

    # TODO add timer
    # TODO check resource usage

    # Load training data
    data = spark.read.options(header=True, inferSchema=True)\
                .csv("../UCI_repo/Concrete/Concrete_Data.csv")
    data.cache()
    print(data.printSchema())
    print(data.describe().toPandas().transpose())

    assembler = VectorAssembler(inputCols=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7',
                                           'f8'], outputCol='features')

    output = assembler.transform(data)

    final_data = output.select('features', 'f9')
    train_data, test_data = final_data.randomSplit([0.75, 0.25])
    test_data.describe().show(3)


    algo = 'linear_regression'
    #algo = 'decision_tree'

    if algo == 'linear_regression':
        from pyspark.ml.regression import LinearRegression
        regressor = LinearRegression(featuresCol = 'features', labelCol='f9',
                                     maxIter=10, regParam=0.3)
    elif algo == 'decision_tree':
        from pyspark.ml.regression import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(featuresCol ='features', labelCol = 'f9')

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
        print("RMSE: %f" % trainingSummary.meanSquaredError)
        print("r2: %f" % trainingSummary.r2)

    # test
    lr_predictions = lr_model.transform(test_data)
    print(lr_predictions.select("prediction", "f9", "features").show(5))
    lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                                       labelCol="f9", metricName="r2")
    print("R Squared (R2) on test data = %g" %lr_evaluator.evaluate(lr_predictions))

    test_result = lr_model.evaluate(test_data)
    print("Root Mean Squared Error (RMSE) on test data = %g" %test_result.meanSquaredError)

    spark.stop()
