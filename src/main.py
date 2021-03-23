from load_dataset import load
from bag_of_words import extract_features
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression


def main():

    print("Create Spark Context ...")
    conf = SparkConf().setAppName("Big Data project")
    sc = SparkContext.getOrCreate(conf=conf)

    print("Create Spark Session ...")
    spark = SparkSession.builder.appName("Big Data project").config("spark.some.config.option", "some-value")\
        .getOrCreate()

    print("Load Dataset ...")
    data_info = load.load_dataset(spark=spark, file_name="../dataset/info_texts.json")
    dataset = load.load_texts(spark=spark, sc=sc, base_path="../dataset", data_info=data_info, split_name='train')

    print("Start Logistic Regression ...")
    tokenizer, hashingTF, idf = extract_features.tf_idf("tweet_text")
    label_stringIdx, dataset = extract_features.transform_labels(dataset)
    dataset = extract_features.create_pipeline(tokenizer, hashingTF, idf, label_stringIdx, dataset)

    print("Logistic Regression ...")
    (trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
    lrModel = lr.fit(trainingData)
    predictions = lrModel.transform(testData)
    predictions.select("id", "labels_str", "tweet_text", "words", "probability", "label", "prediction") \
        .orderBy("probability", ascending=False) \
        .show(n=10, truncate=30)


if __name__ == '__main__':
    main()
