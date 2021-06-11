from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.ml.classification import MultilayerPerceptronClassifier, LogisticRegression, NaiveBayes, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import json


# This function create the model 'Bag-of-Words'
def tf_idf(col_name):

    # Split the tweet text in words and delete punctuation and special characters
    tokenizer = RegexTokenizer(inputCol=col_name, outputCol="words", pattern="\\W")

    # Compute term frequency
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=150)

    # Compute inverse document frequency and remove sparse terms
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms

    return tokenizer, hashingTF, idf


# This function create a single label for each tweet texts
def transform_labels_majority(dataset):

    # Each one of the 150,000 tweets is labeled by 3 different workers using AMT
    # labels: array with 3 numeric labels [0-5] indicating the label by each one of the three AMT annotators
    # 	      0 - NotHate, 1 - Racist, 2 - Sexist, 3 - Homophobe, 4 - Religion, 5 - OtherHate
    # For this project only two labels were considered: Hate and NotHate
    # To determine the label, the three labels of each tweet were checked
    to_single_label = udf(lambda x: 0 if x.count(0) > 1 else 1, IntegerType())

    dataset = dataset.withColumn("label", to_single_label(dataset.labels))

    return dataset


# This function load the json file containing the tweets
def load_dataset(sc, file_name, partitions):

    info_texts = sc.textFile(file_name, partitions)

    data = info_texts.map(lambda x: json.loads(x)).toDF()

    return data


# This function select only the tweets text and
# split the dataset based on the split files
def load_texts(sc, base_path, data_info, split_name, partitions):

    # Upload the file containing a list of tweet texts
    texts = sc.textFile(base_path + "/texts_list.txt", partitions)

    # Upload the split file
    texts_split = sc.textFile(base_path + '/splits/' + split_name + '.txt', partitions).collect()

    # Only tweet text are selected
    texts_list = texts.filter(lambda x: x in texts_split).collect()

    # Split the dataset
    texts_info = data_info.rdd.filter(lambda row: (texts_list.count(row.id) > 0)).toDF()

    return texts_info


def main():

    # Flag
    testMode = False

    print("Create Spark Context ...")
    conf = SparkConf().setAll([("spark.app.name", "Big Data project"),
                               ("spark.ui.showConsoleProgress", "true")])
    sc = SparkContext(conf=conf).getOrCreate()
    sc.setLogLevel("WARN")

    print("Create Spark Session ...")
    spark = SparkSession.builder.appName("Big Data project").getOrCreate()

    # Compute the number of partitions
    partitions = (sc.defaultParallelism * 2)  # (numClusterCores * replicationFactor)

    print("Load Dataset ...")
    dataset = load_dataset(sc=sc, file_name="../dataset/info_texts.json", partitions=partitions)

    print("Split Dataset ...")
    trainingData = load_texts(sc=sc, base_path="../dataset", data_info=dataset, split_name='train', partitions=partitions)
    testData = load_texts(sc=sc, base_path="../dataset", data_info=dataset, split_name='test', partitions=partitions)

    print("Prepare Dataset ...")
    # Prepare test data
    testData = transform_labels_majority(dataset=testData)

    # Prepare training data
    trainingData = transform_labels_majority(dataset=trainingData)

    tokenizer, hashingTF, idf = tf_idf(col_name="tweet_text")

    if testMode:
        #lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
        #nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
        lsvc = LinearSVC(maxIter=10, regParam=0.1)

        pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, lsvc])

        model = pipeline.fit(trainingData)

    else:
        print("Prepare Multilayer Perceptron ...")
        layers = [150, 64, 16, 2]

        # Create the trainer and set its parameters
        ml = MultilayerPerceptronClassifier(maxIter=10, layers=layers, blockSize=128, seed=1234, labelCol="label")

        print("Prepare Cross Validation ...")
        # Define the pipeline for training data
        pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, ml])

        # Define the Param Grid
        paramGrid = ParamGridBuilder().addGrid(ml.solver, ["gd", "l-bfgs"])\
            .addGrid(ml.stepSize, [0.03, 0.02, 0.01]).build()

        crossVal = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid,
                                  evaluator=MulticlassClassificationEvaluator(),
                                  numFolds=5, parallelism=1)

        print("Start Cross Validation ...")
        model = crossVal.fit(trainingData)

    print("Compute Accuracy ...")
    # Compute accuracy on the test set
    result = model.transform(testData)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


if __name__ == '__main__':
    main()
