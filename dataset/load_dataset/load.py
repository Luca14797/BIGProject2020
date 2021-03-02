import json
import os


def load_dataset(spark, file_name):

    data = spark.read.json(file_name, multiLine="true")
    print(data.select("1063020048816660480").show(1, False))

    return data


def load_texts(sc, folder_name):

    texts = os.listdir(folder_name)

    rdd = sc.emptyRDD()

    print("Start create RDD ...")

    for text in texts:

        print("Read file: " + text)

        input = sc.textFile(folder_name + '/' + text)
        data = input.map(lambda x: json.loads(x))

        rdd = rdd.union(data)

    return rdd
