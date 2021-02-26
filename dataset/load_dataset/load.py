import json
import os


def load_dataset(sc, file_name):

    input = sc.textFile(file_name)
    data = input.map(lambda x: json.loads(x))

    return data


def load_texts(sc, folder_name):

    texts = os.listdir(folder_name)

    rdd = sc.emptyRDD()

    print("Start create RDD ...")

    for text in texts:

        print("Read file: " + text)

        input = sc.textFile(folder_name + '/' + text)
        data = input.map(lambda x: json.loads(x))

        rdd.union(data)

    return rdd
