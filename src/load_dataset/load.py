import pyspark.sql.functions as sf
import os


def load_dataset(spark, file_name):

    data = spark.read.json(file_name, multiLine="true")

    return data


def load_texts(spark, sc, base_path, data_info, split_name):

    #texts = os.listdir(base_path + '/texts/')
    texts_split = sc.textFile(base_path + '/splits/' + split_name + '.txt').collect()

    texts = ["1114679353714016256.json", "1063020048816660480.json", "1108927368075374593.json",
             "1114558534635618305.json", "1035252480215592966.json", "1106978219654303744.json",
             "1113920043568463874.json", "1114588617693966336.json", "1045809514740666370.json",
             "1047321356763844608.json", "1107433335318609924.json", "1055298801056014338.json",
             "1110268446623875072.json", "1105839538050646023.json", "1037458903447945216.json",
             "1109072018169839621.json", "1113224872870633472.json", "1114722908297945088.json",
             "1116053548935069697.json", "1116887902859010054.json", "1105630895632011266.json",
             "1115642566853242880.json", "1107491355704475648.json", "1116537005628780544.json",
             "1052191770375835648.json", "1105524709029003267.json", "1105621285655203845.json",
             "1107448953090392066.json", "1108400922198265857.json", "1113991085741826050.json",
             "1115841044514516994.json", "1035111616331825152.json", "1114750222721323008.json",
             "1107692206117318657.json", "1109878793462792193.json", "1058935000215838720.json",
             "1042929715605061632.json", "1116482709474426880.json", "1110025135984005120.json",
             "1053685584118583297.json", "1108208774639165440.json", "1113664242274664451.json",
             "1061668739022815232.json", "1110336630659969024.json", "1114381633614950400.json",
             "1108253229102624768.json", "1035521443927416832.json", "1037466084511830019.json"]

    #texts_info = spark.createDataFrame([], data_info.select(texts_split[0] + '.*').schema)
    texts_info = spark.createDataFrame([], data_info.select("1114679353714016256.*").schema)
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

