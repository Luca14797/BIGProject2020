from dataset.load_dataset import load
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
    print(dataset.show(10, False))


if __name__ == '__main__':
    main()
