# Deep Learning Pipelines for Apache Spark

Deep Learning Pipelines from Databricks provides high-level APIs for scalable deep learning. The
library leverages Spark for its two strongest facets:
1. In the spirit of Spark and Spark MLlib, it provides easy-to-use APIs that enable deep learning
in very few lines of code.
2. It uses Spark's powerful distributed engine to scale out deep learning on massive datasets.

Currently, TensorFlow and TensorFlow-backed Keras workflows are supported, with a focus on model
application and transfer learning on image data at scale, with hyper-parameter tuning in the works.
Furthermore, it provides tools for data scientists and machine learning experts to turn deep
learning models into SQL functions that can be used by a much wider group of people. It does not
perform distributed single-model training - this is an area of active research, and here we aim to
provide the most practical solutions for the vast majority of deep learning use cases.

For an overview of the library, see **[the announcement TODO: put a link to the blog post here]**.
For the various use cases the package serves, see the Quick User Guide section below.

**[TODO: say something about this being an alpha / early iteration, welcoming contributions etc...]**


## Building and running unit tests

To compile this project, run `build/sbt assembly` from the project home directory.
This will also run the Scala unit tests.

To run the Python unit tests, run the `run-tests.sh` script from the `python/` directory.
You will need to set `SPARK_HOME` to your local Spark installation directory, e.g.
```bash
sparkdl$ SPARK_HOME=/usr/local/lib/spark-2.1.1-bin-hadoop2.7 PYSPARK_PYTHON=python2 SCALA_VERSION=2.11.8 SPARK_VERSION=2.1.0 ./python/run-tests.sh
```


## Spark version compatibility

The project supports Spark 2.0 or higher.



## Quick User Guide

The current version of Deep Learning Pipelines provides a suite of tools around working with and
processing images using deep learning. The tools can be categorized as
* [Working with images in Spark](link) : natively in Spark DataFrames
* [Transfer learning](link) : a super quick way to leverage deep learning
* [Applying deep learning models at scale](link) : apply your own or known popular models to image
  data to make predictions or transform them into features
* [Deploying models as SQL functions](link) : empower everyone by making deep learning available in SQL
* Distributed hyper-parameter tuning : via Spark MLlib Pipelines (coming soon)

To try running the examples below, check out the Databricks notebook
[Deep Learning Piplines on Databricks](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/5669198905533692/3647723071348946/3983381308530741/latest.html).


### Working with images in Spark
The first step to applying deep learning on images is the ability to load the images. Deep Learning
Pipelines includes utility functions that can load millions of images into a Spark DataFrame and
decode them automatically in a distributed fashion, allowing manipulationg at scale.

```python
from sparkdl import readImages
image_df = readImages(img_dir)
```

The resulting DataFrame contains a string column named "filePath" containing the path to each image
file, and a image struct ("`SpImage`") column named "image" containing the decoded image data.

```python
image_df.show()
```

The goal is to add support for more data types, such as text and time series, as there is interest.


### Transfer learning
Deep Learning Pipelines provides utilities to perform transfer learning on images, which is one of
the fastest (code and run-time-wise) ways to start using deep learning. Using Deep Learning
Pipelines, it can be done in just several lines of code.


```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])

model = p.fit(train_images_df)    # train_images_df is a dataset of images (SpImage) and labels

# Inspect training error
df = model.transform(train_images_df.limit(10)).select("image", "probability",  "uri", "label")
predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
```


### Applying deep learning models at scale
Spark DataFrames are a natural construct for applying deep learning models to a large-scale dataset.
Deep Learning Pipelines provides a set of (Spark MLlib) Transformers for applying TensorFlow Graphs
and TensorFlow-backed Keras Models at scale. In addition, popular images models can be applied out
of the box, without requiring any TensorFlow or Keras code. The Transformers, backed by the
Tensorframes library, efficiently handle the distribution of models and data to Spark workers.

#### Applying popular image models
There are many well-known deep learning models for images. If the task at hand is very similar to
what the models provide (e.g. object recognition with ImageNet classes), or for pure exploration,
one can use the Transformer `DeepImagePredictor` by simply specifying the model name.

```python
from sparkdl import readImages, DeepImagePredictor

predictor = DeepImagePredictor(inputCol="image", outputCol="predicted_labels",
                               modelName="InceptionV3", decodePredictions=True, topK=10)
image_df = readImages("/data/myimages")
predictions_df = predictor.transform(image_df)
```

#### For TensorFlow users
Deep Learning Pipelines provides a Transformer that will apply the given TensorFlow Graph to a
DataFrame containing a column of images (e.g. loaded using the utilities described in the previous
  section). Here is a very simple example of how a TensorFlow Graph can be used with the
Transformer. In practice, the TensorFlow Graph will likely be restored from files before calling
`TFImageTransformer`.

```python
from sparkdl import readImages, TFImageTransformer
from sparkdl.transformers import utils
import tensorflow as tf

g = tf.Graph()
with g.as_default():
    image_arr = utils.imageInputPlaceholder()
    resized_images = tf.image.resize_images(image_arr, (299, 299))
    # the following step is not necessary for this graph, but can be for graphs with variables, etc
    frozen_graph = utils.stripAndFreezeGraph(g.as_graph_def(add_shapes=True), tf.Session(graph=g),
                                             [resized_images])

transformer = TFImageTransformer(inputCol="image", outputCol="predictions", graph=frozen_graph,
                                 inputTensor=image_arr, outputTensor=resized_images,
                                 outputMode="image")
image_df = readImages("/data/myimages")
processed_image_df = transformer.transform(image_df)
```



#### For Keras users
For applying Keras models in a distributed manner using Spark, [`KerasImageFileTransformer`](link_here)
works on TensorFlow-backed Keras models. It
* Internally creates a DataFrame containing a column of images by applying the user-specified image
loading and processing function to the input DataFrame containing a column of image URIs
* Loads a Keras model from the given model file path
* Applies the model to the image DataFrame

The difference in the API from `TFImageTransformer` above stems from the fact that usual Keras
workflows have very specific ways to load and resize images that are not part of the TensorFlow Graph.


To use the transformer, we first need to have a Keras model stored as a file.

```python
# Save a Keras model file -- for this example we'll just save the InceptionV3 model instead of training one.
from keras.applications import InceptionV3

model = InceptionV3(weights="imagenet")
model.save('/tmp/model-full.h5')
```

Now on the prediction side, we can do:

```python
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from sparkdl import KerasImageFileTransformer

def loadAndPreprocessKerasInceptionV3(uri):
    # this is a typical way to load and prep images in keras
    image = img_to_array(load_img(uri, target_size=(299, 299)))
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)

transformer = KerasImageFileTransformer(inputCol="uri", outputCol="predictions",
                                        modelFile="/tmp/model-full.h5",
                                        imageLoader=loadAndPreprocessKerasInceptionV3,
                                        outputMode="vector")

files = [os.path.abspath(os.path.join(dirpath, f)) for f in os.listdir(img_dir) if f.endswith('.jpg')]
uri_df = sqlContext.createDataFrame(files, StringType()).toDF("uri")

final_df = transformer.transform(uri_df)
```


### Deploying models as SQL functions
Deep learning models often answer questions about complex data, which can be used not only in
production but in interactive analysis of data. SQL is a powerful and widely-used tool for
interactive analysis. By packaging deep learning models as SQL functions, the model developer (you)
can make AI available to a wide range of users.

```python
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sparkdl import registerKerasUDF

def loadAndPreprocessKerasInceptionV3(uri):
    # this is a typical way to load and prep images in keras
    image = img_to_array(load_img(uri, target_size=(299, 299)))
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)

registerKerasUDF("prob_driven_by_007",
                 # model outputs the probability that the photo has 007's car
                 keras_model_file="/mymodels/007model.h5",
                 load_and_preprocess_fn=loadAndPreprocessKerasInceptionV3)
```

Let's say we have image tables we've collected. For this example, we'll register a temp table with
a `SpImage` column.

```python
from sparkdl import readImages
image_df = readImages(img_dir)
image_df.registerTempTable("profile_images")
```

Now any user in SQL can do:

```sql
select image, prob_driven_by_007(image) as p_007
from traffic_imgs
order by prob_driven_by_007(image) DESC
limit 10
```


## Releases:

**[TODO]**
