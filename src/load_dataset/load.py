import pyspark.sql.functions as sf


def load_dataset(spark, file_name):

    data = spark.read.json(file_name, multiLine="true")

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
