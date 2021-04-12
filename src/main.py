from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline

import pyspark.sql.functions as sf
import json


def tf_idf(col_name):

    tokenizer = RegexTokenizer(inputCol=col_name, outputCol="words", pattern="\\W")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=150)
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms

    return tokenizer, hashingTF, idf


def transform_labels(dataset):

    to_single_label = udf(lambda x: 0 if x.count(0) > 1 else 1, IntegerType())
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


def load_texts(spark, sc, base_path, data_info, split_name):

    texts = sc.textFile(base_path + "/texts_list.txt")
    texts_split = sc.textFile(base_path + '/splits/' + split_name + '.txt').collect()

    texts_info = spark.createDataFrame([], data_info.select(texts_split[0] + '.*').schema)
    texts_info = texts_info.withColumn("id", sf.lit(''))
    texts_info = texts_info.select('id', 'img_url', 'labels', 'labels_str', 'tweet_text', 'tweet_url')

    print("Start create DataFrame ...")

    texts_list = texts.filter(lambda x: x in texts_split).collect()

    for text in texts_list:

        text_info = data_info.select(text + ".*")

        text_info = text_info.withColumn("id", sf.lit(text))
        text_info = text_info.select('id', 'img_url', 'labels', 'labels_str', 'tweet_text', 'tweet_url')

        texts_info = texts_info.union(text_info)

    return texts_info


def main():

    print("Create Spark Context ...")
    conf = SparkConf().setAppName("Big Data project")
    sc = SparkContext.getOrCreate(conf=conf)

    print("Create Spark Session ...")
    spark = SparkSession.builder.appName("Big Data project").getOrCreate()

    print("Load Dataset ...")
    dataset = load_dataset(sc=sc, file_name="dataset/info_texts.json")
    #dataset = load_texts(spark=spark, sc=sc, base_path="../dataset", data_info=data_info, split_name='train')

    print("Prepare Logistic Regression ...")
    tokenizer, hashingTF, idf = tf_idf("tweet_text")
    dataset = transform_labels(dataset)
    dataset = create_pipeline(tokenizer, hashingTF, idf, dataset)

    print("Logistic Regression ...")
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=5043)

    layers = [150, 64, 16, 2]

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=10, layers=layers, blockSize=128, seed=1234)

    # train the model
    model = trainer.fit(trainingData)

    # compute accuracy on the test set
    result = model.transform(testData)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    result.select("prediction", "label").show(n=100, truncate=30)
    print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


if __name__ == '__main__':
    main()
