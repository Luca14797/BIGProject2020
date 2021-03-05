import pyspark.sql.functions as sf
import os


def load_dataset(spark, file_name):

    data = spark.read.json(file_name, multiLine="true")

    return data


def load_texts(spark, sc, base_path, data_info, split_name):

    texts = os.listdir(base_path + '/texts/')
    texts_split = sc.textFile(base_path + '/splits' + split_name + '.txt').collect()

    #texts = ["1114679353714016256.json", "1063020048816660480.json", "1106978219654303744.json"]

    texts_info = spark.createDataFrame([], data_info.select(texts_split[0] + '.*').schema)
    texts_info = texts_info.withColumn("id", sf.lit(''))
    texts_info = texts_info.select('id', 'img_url', 'labels', 'labels_str', 'tweet_text', 'tweet_url')

    print("Start create DataFrame ...")

    for text in texts:

        file_name = os.path.splitext(text)[0]

        if file_name in texts_split:

            #print("Read file: " + file_name)

            text_info = data_info.select(file_name + ".*")

            text_info = text_info.withColumn("id", sf.lit(file_name))
            text_info = text_info.select('id', 'img_url', 'labels', 'labels_str', 'tweet_text', 'tweet_url')

            texts_info = texts_info.union(text_info)

    return texts_info

