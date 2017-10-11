import os
from pyspark import SparkContext

from sparkdl.transformers.tf_text import TFTextTransformer

os.environ['PYSPARK_PYTHON'] = '/Users/allwefantasy/python2.7/tensorflow/bin/python'

input_col = "text"
output_col = "preds"

sc = SparkContext.getOrCreate()
documentDF = sc.createDataFrame([
    ("Hi I heard about Spark".split(" "), 1),
    ("I wish Java could use case classes".split(" "), 0),
    ("Logistic regression models are neat".split(" "), 2)
], ["text", "preds"])

transformer = TFTextTransformer(
    inputCol=input_col, outputCol=output_col)

df = transformer.transform(documentDF)
df.show()