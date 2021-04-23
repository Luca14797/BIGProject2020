from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline

import json


def tf_idf(col_name):

    tokenizer = RegexTokenizer(inputCol=col_name, outputCol="words", pattern="\\W")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=150)
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms

    return tokenizer, hashingTF, idf


def transform_labels(dataset):

    to_single_label = udf(lambda x: 0 if x.count(0) > 0 else 1, IntegerType())
    dataset = dataset.withColumn("label", to_single_label(dataset.labels))

    return dataset


def create_pipeline(tokenizer, hashingTF, idf, dataset):

    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])
    pipelineFit = pipeline.fit(dataset)
    dataset = pipelineFit.transform(dataset)

    return dataset


def load_dataset(sc, file_name):

    info_texts = sc.textFile(file_name)

    data = info_texts.map(lambda x: json.loads(x)).toDF()

    return data


def load_texts(sc, base_path, data_info, split_name):

    texts = sc.textFile(base_path + "/texts_list.txt")
    texts_split = sc.textFile(base_path + '/splits/' + split_name + '.txt').collect()

    texts_list = texts.filter(lambda x: x in texts_split).collect()

    texts_info = data_info.filter(data_info.id.isin(texts_list))

    return texts_info


def main():

    print("Create Spark Context ...")
    conf = SparkConf().setAppName("Big Data project")
    sc = SparkContext.getOrCreate(conf=conf)

    print("Create Spark Session ...")
    spark = SparkSession.builder.appName("Big Data project").getOrCreate()

    print("Load Dataset ...")
    dataset = load_dataset(sc=sc, file_name="dataset/info_texts.json")

    print("Split Dataset ...")
    trainingData = load_texts(sc=sc, base_path="dataset", data_info=dataset, split_name='train')
    testData = load_texts(sc=sc, base_path="dataset", data_info=dataset, split_name='test')

    print("Prepare Multilayer Perceptron ...")
    # prepare training data
    tokenizer, hashingTF, idf = tf_idf("tweet_text")
    trainingData = transform_labels(trainingData)
    trainingData = create_pipeline(tokenizer, hashingTF, idf, trainingData)

    # prepare test data
    tokenizer, hashingTF, idf = tf_idf("tweet_text")
    testData = transform_labels(testData)
    testData = create_pipeline(tokenizer, hashingTF, idf, testData)

    print("Multilayer Perceptron Training ...")

    layers = [150, 64, 16, 2]

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=10, layers=layers, blockSize=128, seed=1234)

    # train the model
    model = trainer.fit(trainingData)

    print("Compute Accuracy ...")
    # compute accuracy on the test set
    result = model.transform(testData)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


if __name__ == '__main__':
    main()
