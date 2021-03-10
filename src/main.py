from load_dataset import load
from bag_of_words import extract_frequency
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType


def main():

    print("Create Spark Context")
    #conf = SparkConf().setMaster("spark://DESKTOP-HTII7QG.localdomain:7077").setAppName("My App")
    conf = SparkConf().setMaster("local").setAppName("My App")
    sc = SparkContext(conf=conf)

    '''
    spark = SparkSession.builder.master("spark://DESKTOP-HTII7QG.localdomain:7077").appName("My App")\
        .config("spark.some.config.option", "some-value").getOrCreate()
    '''
    spark = SparkSession.builder.master("local").appName("My App").config("spark.some.config.option", "some-value")\
        .getOrCreate()

    print("Load dataset")
    data_info = load.load_dataset(spark=spark, file_name="../dataset/prova.json")
    dataset = load.load_texts(spark=spark, sc=sc, base_path="../dataset", data_info=data_info, split_name='train')
    '''
    wordsData = extract_frequency.extract_words(data=dataset, col_name='tweet_text')
    featurizedData = extract_frequency.extract_feature(wordsData=wordsData)
    rescaledData = extract_frequency.extract_frequency(featurizedData=featurizedData)
    '''

    tokenizer = Tokenizer(inputCol="tweet_text", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms
    to_string_udf = udf(lambda x: ''.join(str(e) for e in x), StringType())
    dataset = dataset.withColumn("labelstring", to_string_udf(dataset.labels))
    label_stringIdx = StringIndexer(inputCol="labelstring", outputCol="label")
    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, label_stringIdx])
    pipelineFit = pipeline.fit(dataset)
    dataset = pipelineFit.transform(dataset)
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testData)
    predictions.select("id", "labels_str", "tweet_text", "probability", "label", "prediction") \
        .orderBy("probability", ascending=False) \
        .show(n=10, truncate=30)


if __name__ == '__main__':
    main()
