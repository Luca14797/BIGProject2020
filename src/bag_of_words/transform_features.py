from pyspark.ml.feature import MinMaxScaler


def features_scaler(data):

    scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

    scalerModel = scaler.fit(data)

    scaledData = scalerModel.transform(data)

    return scaledData
