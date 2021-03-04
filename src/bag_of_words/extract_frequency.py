from pyspark.ml.feature import HashingTF, IDF, Tokenizer


def extract_words(data, col_name):

    tokenizer = Tokenizer(inputCol=col_name, outputCol="words")
    wordsData = tokenizer.transform(data)

    return wordsData


def extract_feature(wordsData):

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=100000000)
    featurizedData = hashingTF.transform(wordsData)

    return featurizedData


def extract_frequency(featurizedData):

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    return rescaledData
