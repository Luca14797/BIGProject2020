from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline


def tf_idf(col_name):

    tokenizer = RegexTokenizer(inputCol=col_name, outputCol="words", pattern="\\W")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms

    return tokenizer, hashingTF, idf


def transform_labels(dataset):

    to_string_udf = udf(lambda x: ''.join(str(e) for e in x), StringType())
    dataset = dataset.withColumn("labelstring", to_string_udf(dataset.labels))
    label_stringIdx = StringIndexer(inputCol="labelstring", outputCol="label")

    return label_stringIdx, dataset


def create_pipeline(tokenizer, hashingTF, idf, label_stringIdx, dataset):

    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, label_stringIdx])
    pipelineFit = pipeline.fit(dataset)
    dataset = pipelineFit.transform(dataset)

    return dataset
