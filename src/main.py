from dataset.load_dataset import load
from src.bag_of_words import extract_frequency
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


def main():

    print("Create Spark Context")
    conf = SparkConf().setMaster("local").setAppName("My App")
    sc = SparkContext(conf=conf)

    spark = SparkSession.builder.appName("My App").config("spark.some.config.option", "some-value").getOrCreate()

    print("Load dataset")
    data_info = load.load_dataset(spark=spark, file_name="../dataset/prova.json")
    dataset = load.load_texts(spark=spark, folder_name="../dataset/texts", data_info=data_info)

    wordsData = extract_frequency.extract_words(data=dataset, col_name='tweet_text')
    featurizedData = extract_frequency.extract_feature(wordsData=wordsData)
    rescaledData = extract_frequency.extract_frequency(featurizedData=featurizedData)


if __name__ == '__main__':
    main()
