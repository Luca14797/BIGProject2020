from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, StructType, StructField, ArrayType, LongType
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

import pyspark.sql.functions as sf
import json


def tf_idf(col_name):

    tokenizer = RegexTokenizer(inputCol=col_name, outputCol="words", pattern="\\W")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms

    return tokenizer, hashingTF, idf


def transform_labels(dataset):

    to_string_udf = udf(lambda x: ''.join(str(e) for e in x), StringType())
    dataset = dataset.withColumn("labelstring", to_string_udf(dataset.labels))
    label_stringIdx = StringIndexer(inputCol="labelstring", outputCol="label")

    return label_stringIdx, dataset


def create_pipeline(tokenizer, hashingTF, idf, label_stringIdx, dataset):

    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, label_stringIdx])
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
    dataset = load_dataset(sc=sc, file_name="../dataset/new_info_texts.json")
    #dataset = load_texts(spark=spark, sc=sc, base_path="../dataset", data_info=data_info, split_name='train')

    print("Start Logistic Regression ...")
    tokenizer, hashingTF, idf = tf_idf("tweet_text")
    label_stringIdx, dataset = transform_labels(dataset)
    dataset = create_pipeline(tokenizer, hashingTF, idf, label_stringIdx, dataset)

    print("Logistic Regression ...")
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testData)
    predictions.select("id", "labels_str", "tweet_text", "words", "probability", "label", "prediction") \
        .orderBy("probability", ascending=False) \
        .show(n=100, truncate=30)


if __name__ == '__main__':
    main()
