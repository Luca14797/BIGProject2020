import pyspark.sql.functions as sf
import os


def load_dataset(spark, file_name):

    data = spark.read.json(file_name, multiLine="true")

    return data


def load_texts(spark, folder_name, data_info):

    #texts = os.listdir(folder_name)
    texts = ["1114679353714016256.json", "1063020048816660480.json", "1106978219654303744.json"]

    texts_info = spark.createDataFrame([], data_info.select("1035252480215592966.*").schema)
    texts_info = texts_info.withColumn("id", sf.lit(''))
    texts_info = texts_info.select('id', 'img_url', 'labels', 'labels_str', 'tweet_text', 'tweet_url')

    print("Start create DataFrame ...")

    for text in texts:

        file_name = os.path.splitext(text)[0]

        #print("Read file: " + file_name)

        text_info = data_info.select(file_name + ".*")

        text_info = text_info.withColumn("id", sf.lit(file_name))
        text_info = text_info.select('id', 'img_url', 'labels', 'labels_str', 'tweet_text', 'tweet_url')

        texts_info = texts_info.union(text_info)

    return texts_info
