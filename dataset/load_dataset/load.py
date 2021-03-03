import json
import os


def load_dataset(spark, file_name):

    data = spark.read.json(file_name, multiLine="true")
    print(data.show(10, False))

    return data


def load_texts(spark, folder_name, data_info):

    #texts = os.listdir(folder_name)
    texts = ["1114679353714016256.json", "1063020048816660480.json", "1106978219654303744.json"]

    texts_info = None

    print("Start create DataFrame ...")

    for text in texts:

        file_name = os.path.splitext(text)[0]

        print("Read file: " + file_name)

        text_info = data_info.select(file_name)

        if texts_info is None:

            texts_info = text_info

        else:

            texts_info = texts_info.union(text_info)

    return texts_info
