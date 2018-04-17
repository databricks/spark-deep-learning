Deep Learning Pipelines for Apache Spark
============================================================
[![Build Status][pkg-build-badge]][pkg-build-link] [![Coverage][pkg-cov-badge]][pkg-cov-link]

  [pkg-build-badge]: https://travis-ci.org/databricks/spark-deep-learning.svg?branch=master
  [pkg-build-link]: https://travis-ci.org/databricks/spark-deep-learning
  [pkg-cov-badge]: https://codecov.io/gh/databricks/spark-deep-learning/coverage.svg?branch=master
  [pkg-cov-link]: https://codecov.io/gh/databricks/spark-deep-learning/branch/master

Deep Learning Pipelines provides high-level APIs for scalable deep learning in Python with Apache Spark.

- [Overview](#overview)
- [Building and running unit tests](#building-and-running-unit-tests)
- [Spark version compatibility](#spark-version-compatibility)
- [Support](#support)
- [Releases](#releases)
- [Quick user guide](#quick-user-guide)
  - [Working with images in Spark](#working-with-images-in-spark)
  - [Transfer learning](#transfer-learning)
  - [Applying deep learning models at scale](#applying-deep-learning-models-at-scale)
  - [Deploying models as SQL functions](#deploying-models-as-sql-functions)
- [License](#license)


## Overview

Deep Learning Pipelines provides high-level APIs for scalable deep learning in Python with Apache Spark.

The library comes from Databricks and leverages Spark for its two strongest facets:

1.  In the spirit of Spark and [Spark MLlib](https://spark.apache.org/mllib/), it provides easy-to-use APIs that enable deep learning in very few lines of code.
2.  It uses Spark's powerful distributed engine to scale out deep learning on massive datasets.

Currently, TensorFlow and TensorFlow-backed Keras workflows are supported, with a focus on:
* large-scale inference / scoring
* transfer learning and hyperparameter tuning on image data

Furthermore, it provides tools for data scientists and machine learning experts to turn deep learning models into SQL functions that can be used by a much wider group of users. It does not perform single-model distributed training - this is an area of active research, and here we aim to provide the most practical solutions for the majority of deep learning use cases.

For an overview of the library, see the Databricks [blog post](https://databricks.com/blog/2017/06/06/databricks-vision-simplify-large-scale-deep-learning.html?preview=true) introducing Deep Learning Pipelines. For the various use cases the package serves, see the [Quick user guide](#quick-user-guide) section below.

The library is in its early days, and we welcome everyone's feedback and contribution.

Maintainers: Bago Amirbekian, Joseph Bradley, Sue Ann Hong, Tim Hunter, Siddharth Murching, Tomas Nykodym


## Building and running unit tests

To compile this project, run `build/sbt assembly` from the project home directory. This will also run the Scala unit tests.

To run the Python unit tests, run the `run-tests.sh` script from the `python/` directory (after compiling). You will need to set a few environment variables, e.g.

```bash
# Be sure to run build/sbt assembly before running the Python tests
sparkdl$ SPARK_HOME=/usr/local/lib/spark-2.1.1-bin-hadoop2.7 PYSPARK_PYTHON=python2 SCALA_VERSION=2.11.8 SPARK_VERSION=2.1.1 ./python/run-tests.sh
```

## Spark version compatibility

Spark 2.2.0 and Python 3.6 are recommended for working with the latest code. See the [travis config](https://github.com/databricks/spark-deep-learning/blob/master/.travis.yml) for the regularly-tested combinations.

Compatibility requirements for each release are listed in the [Releases](#releases) section.


## Support

You can ask questions and join the development discussion on the [DL Pipelines Google group](https://groups.google.com/forum/#!forum/dl-pipelines-users/).

You can also post bug reports and feature requests in Github issues.


## Releases
<!--
TODO: might want to add TensorFlow compatibility information.
- 1.0.0 release: Spark 2.3 required. Python 3.6 & Scala 2.11 recommended. TensorFlow 1.5.0+ required.
    1. Using the definition of images from Spark 2.3. The new definition uses the BGR channel ordering 
       for 3-channel images instead of the RGB ordering used in this project before the change. 
    2. Persistence for DeepImageFeaturizer (both Python and Scala).
-->
- [0.3.0](https://github.com/databricks/spark-deep-learning/releases/tag/v0.3.0) release: Spark 2.2.0, Python 3.6 & Scala 2.11 recommended. TensorFlow 1.4.1- required.
    1. KerasTransformer & TFTransformer for large-scale batch inference on non-image (tensor) data.
    2. Scala API for transfer learning (`DeepImageFeaturizer`). InceptionV3 is supported.
    3. Added VGG16, VGG19 models to DeepImageFeaturizer & DeepImagePredictor (Python).
- [0.2.0](https://github.com/databricks/spark-deep-learning/releases/tag/v0.2.0) release: Spark 2.1.1 & Python 2.7 recommended.
    1. KerasImageFileEstimator API (train a Keras model on image files)
    2. SQL UDF support for Keras models
    3. Added Xception, Resnet50 models to DeepImageFeaturizer & DeepImagePredictor.
- 0.1.0 Alpha release: Spark 2.1.1 & Python 2.7 recommended.


## Quick user guide

Deep Learning Pipelines provides a suite of tools around working with and processing images using deep learning. The tools can be categorized as

-   [Working with images in Spark](#working-with-images-in-spark) : natively in Spark DataFrames
-   [Transfer learning](#transfer-learning) : a super quick way to leverage deep learning
-   Distributed hyper-parameter tuning : via Spark MLlib Pipelines (coming soon)
-   [Applying deep learning models at scale - to images](#applying-deep-learning-models-at-scale) : apply your own or known popular models to make predictions or transform them into features
-   [Applying deep learning models at scale - to tensors](#applying-deep-learning-models-at-scale-to-tensors) : of up to 2 dimensions
-   [Deploying models as SQL functions](#deploying-models-as-sql-functions) : empower everyone by making deep learning available in SQL.

To try running the examples below, check out the Databricks notebook in the [Databricks docs for Deep Learning Pipelines](https://docs.databricks.com/applications/deep-learning/deep-learning-pipelines.html), which works with the latest release of Deep Learning Pipelines. Here are some Databricks notebooks compatible with earlier releases:
[0.1.0](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/5669198905533692/3647723071348946/3983381308530741/latest.html),
[0.2.0](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/5669198905533692/1674891575666800/3983381308530741/latest.html),
[0.3.0](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/4856334613426202/3381529530484660/4079725938146156/latest.html).


### Working with images in Spark

The first step to applying deep learning on images is the ability to load the images. Spark and Deep Learning Pipelines include utility functions that can load millions of images into a Spark DataFrame and decode them automatically in a distributed fashion, allowing manipulation at scale.

Using Spark's ImageSchema

```python
from pyspark.ml.image import ImageSchema
image_df = ImageSchema.readImages("/data/myimages")
```

or if custom image library is needed:

```python
from sparkdl.image import imageIO as imageIO
image_df = imageIO.readImagesWithCustomFn("/data/myimages",decode_f=<your image library, see imageIO.PIL_decode>)
```

The resulting DataFrame contains a string column named "image" containing an image struct with schema == ImageSchema.

```python
image_df.show()
```

**Why images?** Deep learning has shown to be powerful for tasks involving images, so we have added native Spark support for images. The goal is to support more data types, such as text and time series, based on community interest.


### Transfer learning

Deep Learning Pipelines provides utilities to perform [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) on images, which is one of the fastest (code and run-time-wise) ways to start using deep learning. Using Deep Learning Pipelines, it can be done in just several lines of code.

```python
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])

model = p.fit(train_images_df)    # train_images_df is a dataset of images and labels

# Inspect training error
df = model.transform(train_images_df.limit(10)).select("image", "probability",  "uri", "label")
predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
```

### Applying deep learning models at scale

Spark DataFrames are a natural construct for applying deep learning models to a large-scale dataset. Deep Learning Pipelines provides a set of Spark MLlib Transformers for applying TensorFlow Graphs and TensorFlow-backed Keras Models at scale. The Transformers, backed by the Tensorframes library, efficiently handle the distribution of models and data to Spark workers.

#### Applying deep learning models at scale to images
Deep Learning Pipelines provides several ways to apply models to images at scale: 
* Popular images models can be applied out of the box, without requiring any TensorFlow or Keras code
* TensorFlow graphs that work on images
* Keras models that work on images

##### Applying popular image models
There are many well-known deep learning models for images. If the task at hand is very similar to what the models provide (e.g. object recognition with ImageNet classes), or for pure exploration, one can use the Transformer `DeepImagePredictor` by simply specifying the model name.


```python
from sparkdl import readImages, DeepImagePredictor

image_df = readImages(sample_img_dir)

predictor = DeepImagePredictor(inputCol="image", outputCol="predicted_labels", modelName="InceptionV3", decodePredictions=True, topK=10)
predictions_df = predictor.transform(image_df)
```

##### For TensorFlow users
Deep Learning Pipelines provides an MLlib Transformer that will apply the given TensorFlow Graph to a DataFrame containing a column of images (e.g. loaded using the utilities described in the previous section). Here is a very simple example of how a TensorFlow Graph can be used with the Transformer. In practice, the TensorFlow Graph will likely be restored from files before calling `TFImageTransformer`.


```python
from sparkdl import readImages, TFImageTransformer
import sparkdl.graph.utils as tfx  # strip_and_freeze_until was moved from sparkdl.transformers to sparkdl.graph.utils in 0.2.0
from sparkdl.transformers import utils
import tensorflow as tf

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    image_arr = utils.imageInputPlaceholder()
    resized_images = tf.image.resize_images(image_arr, (299, 299))
    # the following step is not necessary for this graph, but can be for graphs with variables, etc
    frozen_graph = tfx.strip_and_freeze_until([resized_images], graph, sess, return_graph=True)

transformer = TFImageTransformer(inputCol="image", outputCol="predictions", graph=frozen_graph,
                                 inputTensor=image_arr, outputTensor=resized_images,
                                 outputMode="image")

image_df = readImages(sample_img_dir)
processed_image_df = transformer.transform(image_df)
```

##### For Keras users
For applying Keras models in a distributed manner using Spark, [`KerasImageFileTransformer`](link_here) works on TensorFlow-backed Keras models. It 
* Internally creates a DataFrame containing a column of images by applying the user-specified image loading and processing function to the input DataFrame containing a column of image URIs
* Loads a Keras model from the given model file path 
* Applies the model to the image DataFrame

The difference in the API from `TFImageTransformer` above stems from the fact that usual Keras workflows have very specific ways to load and resize images that are not part of the TensorFlow Graph.

To use the transformer, we first need to have a Keras model stored as a file. For this notebook we'll just save the Keras built-in InceptionV3 model instead of training one.


```python
from keras.applications import InceptionV3

model = InceptionV3(weights="imagenet")
model.save('/tmp/model-full.h5')
```

Now on the prediction side:


```python
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from pyspark.sql.types import StringType
from sparkdl import KerasImageFileTransformer

def loadAndPreprocessKerasInceptionV3(uri):
  # this is a typical way to load and prep images in keras
  image = img_to_array(load_img(uri, target_size=(299, 299)))  # image dimensions for InceptionV3
  image = np.expand_dims(image, axis=0)
  return preprocess_input(image)

transformer = KerasImageFileTransformer(inputCol="uri", outputCol="predictions",
                                        modelFile='/tmp/model-full-tmp.h5',  # local file path for model
                                        imageLoader=loadAndPreprocessKerasInceptionV3,
                                        outputMode="vector")

files = [os.path.abspath(os.path.join(dirpath, f)) for f in os.listdir("/data/myimages") if f.endswith('.jpg')]
uri_df = sqlContext.createDataFrame(files, StringType()).toDF("uri")

keras_pred_df = transformer.transform(uri_df)
```


#### Applying deep learning models at scale to tensors
Deep Learning Pipelines also provides ways to apply models with tensor inputs (up to 2 dimensions), written in popular deep learning libraries:
* TensorFlow graphs
* Keras models

##### For TensorFlow users

`TFTransformer` applies a user-specified TensorFlow graph to tensor inputs of up to 2 dimensions.
The TensorFlow graph may be specified as TensorFlow graph objects (`tf.Graph` or `tf.GraphDef`) or checkpoint or `SavedModel` objects
(see the [input object class](https://github.com/databricks/spark-deep-learning/blob/master/python/sparkdl/graph/input.py#L27) for more detail).
The `transform()` function applies the TensorFlow graph to a column of arrays (where an array corresponds to a Tensor) in the input DataFrame
and outputs a column of arrays corresponding to the output of the graph.

First we generate sample dataset of 2-dimensional points, Gaussian distributed around two different centers

```python
import numpy as np
from pyspark.sql.types import Row

n_sample = 1000
center_0 = [-1.5, 1.5]
center_1 = [1.5, -1.5]

def to_row(args):
  xy, l = args
  return Row(inputCol = xy, label = l)

samples_0 = [np.random.randn(2) + center_0 for _ in range(n_sample//2)]
labels_0 = [0 for _ in range(n_sample//2)]
samples_1 = [np.random.randn(2) + center_1 for _ in range(n_sample//2)]
labels_1 = [1 for _ in range(n_sample//2)]

rows = map(to_row, zip(map(lambda x: x.tolist(), samples_0 + samples_1), labels_0 + labels_1))
sdf = spark.createDataFrame(rows)

```

Next, we write a function that returns a tensorflow graph and its input

```python
import tensorflow as tf

def build_graph(sess, w0):
  X = tf.placeholder(tf.float32, shape=[None, 2], name="input_tensor")
  model = tf.sigmoid(tf.matmul(X, w0), name="output_tensor")
  return model, X

```

Following is the code you would write to predict using tensorflow on a single node.

```python
w0 = np.array([[1], [-1]]).astype(np.float32)
with tf.Session() as sess:
  model, X = build_graph(sess, w0)
  output = sess.run(model, feed_dict = {
    X : samples_0 + samples_1
  })
```

Now you can use the following Spark MLlib Transformer to apply the model to a DataFrame in a distributed fashion.

```python
from sparkdl import TFTransformer
from sparkdl.graph.input import TFInputGraph
import sparkdl.graph.utils as tfx

graph = tf.Graph()
with tf.Session(graph=graph) as session, graph.as_default():
    _, _ = build_graph(session, w0)
    gin = TFInputGraph.fromGraph(session.graph, session,
                                 ["input_tensor"], ["output_tensor"])

transformer = TFTransformer(
    tfInputGraph=gin,
    inputMapping={'inputCol': tfx.tensor_name("input_tensor")},
    outputMapping={tfx.tensor_name("output_tensor"): 'outputCol'})

odf = transformer.transform(sdf)
```

##### For Keras users
`KerasTransformer` applies a TensorFlow-backed Keras model to tensor inputs of up to 2 dimensions. It loads a Keras model from a given model file path and applies the model to a column of arrays (where an array corresponds to a Tensor), outputting a column of arrays.


```python
from sparkdl import KerasTransformer
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Generate random input data
num_features = 10
num_examples = 100
input_data = [{"features" : np.random.randn(num_features).tolist()} for i in range(num_examples)]
input_df = sqlContext.createDataFrame(input_data)

# Create and save a single-hidden-layer Keras model for binary classification
# NOTE: In a typical workflow, we'd train the model before exporting it to disk,
# but we skip that step here for brevity
model = Sequential()
model.add(Dense(units=20, input_shape=[num_features], activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model_path = "/tmp/simple-binary-classification"
model.save(model_path)

# Create transformer and apply it to our input data
transformer = KerasTransformer(inputCol="features", outputCol="predictions", modelFile=model_path)
final_df = transformer.transform(input_df)
```

### Deploying models as SQL functions

One way to productionize a model is to deploy it as a Spark SQL User Defined Function, which allows anyone who knows SQL to use it. Deep Learning Pipelines provides mechanisms to take a deep learning model and *register* a Spark SQL User Defined Function (UDF). In particular, Deep Learning Pipelines 0.2.0 adds support for creating SQL UDFs from Keras models that work on image data. 

The resulting UDF takes a column (formatted as a image struct "SpImage") and produces the output of the given Keras model; e.g. for Inception V3, it produces a real valued score vector over the ImageNet object categories.

We can register a UDF for a Keras model that works on images as follows:

```python
from keras.applications import InceptionV3
from sparkdl.udf.keras_image_model import registerKerasImageUDF

registerKerasImageUDF("inceptionV3_udf", InceptionV3(weights="imagenet"))
```

Alternatively, we can also register a UDF from a model file:

```python
registerKerasImageUDF("my_custom_keras_model_udf", "/tmp/model-full-tmp.h5")
```

In Keras workflows dealing with images, it's common to have preprocessing steps before the model is applied to the image. If our workflow requires preprocessing, we can optionally provide a preprocessing function to UDF registration. The preprocessor should take in a filepath and return an image array; below is a simple example.

```python
from keras.applications import InceptionV3
from sparkdl.udf.keras_image_model import registerKerasImageUDF

def keras_load_img(fpath):
    from keras.preprocessing.image import load_img, img_to_array
    import numpy as np
    img = load_img(fpath, target_size=(299, 299))
    return img_to_array(img).astype(np.uint8)

registerKerasImageUDF("inceptionV3_udf_with_preprocessing", InceptionV3(weights="imagenet"), keras_load_img)
```

Once a UDF has been registered, it can be used in a SQL query, e.g.

```python
from sparkdl import readImages

image_df = readImages(sample_img_dir)
image_df.registerTempTable("sample_images")
```

```sql
SELECT my_custom_keras_model_udf(image) as predictions from sample_images
```


## License
* The Deep Learning Pipelines source code is released under the Apache License 2.0 (see the LICENSE file).
* Models marked as *provided by Keras* (used by `DeepImageFeaturizer` and `DeepImagePredictor`) are provided subject to the MIT license located at https://github.com/fchollet/keras/blob/master/LICENSE and subject to any additional copyrights and licenses specified in the code or documentation. Also see the [Keras applications page](https://keras.io/applications/) for more on the individual model licensing information.
