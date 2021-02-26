from dataset.load_dataset import load
from pyspark import SparkContext, SparkConf


def main():

    print("Create Spark Context")
    conf = SparkConf().setMaster("local").setAppName("My App")
    sc = SparkContext(conf=conf)

    print("Load dataset")
    dataset = load.load_texts(sc=sc, folder_name="../dataset/texts")

    print(dataset.take(1))


if __name__ == '__main__':
    main()
