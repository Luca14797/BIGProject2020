from load_dataset import load
from bag_of_words import extract_features
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression


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

    tokenizer, hashingTF, idf = extract_features.tf_idf("tweet_text")
    label_stringIdx, dataset = extract_features.transform_labels(dataset)
    dataset = extract_features.create_pipeline(tokenizer, hashingTF, idf, label_stringIdx, dataset)

    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testData)
    predictions.select("id", "labels_str", "tweet_text", "words", "probability", "label", "prediction") \
        .orderBy("probability", ascending=False) \
        .show(n=10, truncate=30)


if __name__ == '__main__':
    main()
