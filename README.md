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
- [Quick user guide](#quick-user-guide)
  - [Working with images in Spark](#working-with-images-in-spark)
  - [Transfer learning](#transfer-learning)
  - [Applying deep learning models at scale](#applying-deep-learning-models-at-scale)
  - [Deploying models as SQL functions](#deploying-models-as-sql-functions)
- [Releases](#releases)

## Overview

Deep Learning Pipelines provides high-level APIs for scalable deep learning in Python with Apache Spark.

The library comes from Databricks and leverages Spark for its two strongest facets:

1.  In the spirit of Spark and [Spark MLlib](https://spark.apache.org/mllib/), it provides easy-to-use APIs that enable deep learning in very few lines of code.
2.  It uses Spark's powerful distributed engine to scale out deep learning on massive datasets.

Currently, TensorFlow and TensorFlow-backed Keras workflows are supported, with a focus on model inference/scoring and transfer learning on image data at scale, with hyper-parameter tuning in the works.

Furthermore, it provides tools for data scientists and machine learning experts to turn deep learning models into SQL functions that can be used by a much wider group of users. It does not perform single-model distributed training - this is an area of active research, and here we aim to provide the most practical solutions for the majority of deep learning use cases.

For an overview of the library, see the Databricks [blog post](https://databricks.com/blog/2017/06/06/databricks-vision-simplify-large-scale-deep-learning.html?preview=true) introducing Deep Learning Pipelines. For the various use cases the package serves, see the [Quick user guide](#quick-user-guide) section below.

The library is in its early days, and we welcome everyone's feedback and contribution.

Maintainers: Bago Amirbekian, Joseph Bradley, Sue Ann Hong, Tim Hunter, Philip Yang


## Building and running unit tests

To compile this project, run `build/sbt assembly` from the project home directory. This will also run the Scala unit tests.

To run the Python unit tests, run the `run-tests.sh` script from the `python/` directory. You will need to set a few environment variables, e.g.

```bash
# Be sure to run build/sbt assembly before running the Python tests
sparkdl$ SPARK_HOME=/usr/local/lib/spark-2.1.1-bin-hadoop2.7 PYSPARK_PYTHON=python2 SCALA_VERSION=2.11.8 SPARK_VERSION=2.1.1 ./python/run-tests.sh
```


## Spark version compatibility

Spark 2.1.1 and Python 2.7 are recommended.


## Quick user guide

The current version of Deep Learning Pipelines provides a suite of tools around working with and processing images using deep learning. The tools can be categorized as

-   [Working with images in Spark](#working-with-images-in-spark) : natively in Spark DataFrames
-   [Transfer learning](#transfer-learning) : a super quick way to leverage deep learning
-   [Applying deep learning models at scale](#applying-deep-learning-models-at-scale) : apply your own or known popular models to image data to make predictions or transform them into features
-   [Deploying models as SQL functions](#deploying-models-as-sql-functions) : empower everyone by making deep learning available in SQL.
-   Distributed hyper-parameter tuning : via Spark MLlib Pipelines (coming soon)

To try running the examples below, check out the Databricks notebook [DeepLearning Pipelines on Databricks](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/5669198905533692/3647723071348946/3983381308530741/latest.html).

### Working with images in Spark

The first step to applying deep learning on images is the ability to load the images. Deep Learning Pipelines includes utility functions that can load millions of images into a Spark DataFrame and decode them automatically in a distributed fashion, allowing manipulation at scale.

```python
from sparkdl import readImages
image_df = readImages("/data/myimages")
```

The resulting DataFrame contains a string column named "filePath" containing the path to each image file, and a image struct ("`SpImage`") column named "image" containing the decoded image data.

```python
image_df.show()
```

The goal is to add support for more data types, such as text and time series, as there is interest.

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

model = p.fit(train_images_df)    # train_images_df is a dataset of images (SpImage) and labels

# Inspect training error
df = model.transform(train_images_df.limit(10)).select("image", "probability",  "uri", "label")
predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
```

### Applying deep learning models at scale

Spark DataFrames are a natural construct for applying deep learning models to a large-scale dataset. Deep Learning Pipelines provides a set of (Spark MLlib) Transformers for applying TensorFlow Graphs and TensorFlow-backed Keras Models at scale. In addition, popular images models can be applied out of the box, without requiring any TensorFlow or Keras code. The Transformers, backed by the Tensorframes library, efficiently handle the distribution of models and data to Spark workers.

1.  Applying popular image models

    There are many well-known deep learning models for images. If the task at hand is very similar to what the models provide (e.g. object recognition with ImageNet classes), or for pure exploration, one can use the Transformer `DeepImagePredictor` by simply specifying the model name.

    ```python
    from sparkdl import readImages, DeepImagePredictor

    predictor = DeepImagePredictor(inputCol="image", outputCol="predicted_labels",
                                   modelName="InceptionV3", decodePredictions=True, topK=10)
    image_df = readImages("/data/myimages")
    predictions_df = predictor.transform(image_df)
    ```

2.  For TensorFlow users

    Deep Learning Pipelines provides a Transformer that will apply the given TensorFlow Graph to a DataFrame containing a column of images (e.g. loaded using the utilities described in the previous section). Here is a very simple example of how a TensorFlow Graph can be used with the Transformer. In practice, the TensorFlow Graph will likely be restored from files before calling `TFImageTransformer`.

    ```python
    from sparkdl import readImages, TFImageTransformer
    import sparkdl.graph.utils as tfx
    from sparkdl.transformers import utils
    import tensorflow as tf

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        image_arr = utils.imageInputPlaceholder()
        resized_images = tf.image.resize_images(image_arr, (299, 299))
        frozen_graph = tfx.strip_and_freeze_until([resized_images], graph, sess,
                                                  return_graph=True)

    transformer = TFImageTransformer(inputCol="image", outputCol="predictions", graph=frozen_graph,
                                     inputTensor=image_arr, outputTensor=resized_images,
                                     outputMode="image")
    image_df = readImages("/data/myimages")
    processed_image_df = transformer.transform(image_df)
    ```

3.  For Keras users

    For applying Keras models in a distributed manner using Spark, `KerasImageFileTransformer` works on TensorFlow-backed Keras models. It

    1.  Internally creates a DataFrame containing a column of images by applying the user-specified image loading and processing function to the input DataFrame containing a column of image URIs
    2.  Loads a Keras model from the given model file path
    3.  Applies the model to the image DataFrame

    The difference in the API from `TFImageTransformer` above stems from the fact that usual Keras workflows have very specific ways to load and resize images that are not part of the TensorFlow Graph.

    To use the transformer, we first need to have a Keras model stored as a file. For this example we'll just save the Keras built-in InceptionV3 model instead of training one.

    ```python
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

    files = [os.path.abspath(os.path.join(dirpath, f)) for f in os.listdir("/data/myimages") if f.endswith('.jpg')]
    uri_df = sqlContext.createDataFrame(files, StringType()).toDF("uri")

    final_df = transformer.transform(uri_df)
    ```


### Deploying models as SQL functions

One way to productionize a model is to deploy it as a [Spark SQL User Defined Function](https://docs.databricks.com/spark/latest/spark-sql/udf-in-python.html), which allows anyone who knows SQL to use it. Deep Learning Pipelines provides mechanisms to take a deep learning model and register a Spark SQL User Defined Function (UDF).

The resulting UDF takes a column (formatted as a image struct "`SpImage`") and produces the output of the given Keras model (e.g. for [Inception V3](https://keras.io/applications/#inceptionv3), it produces a real valued score vector over the ImageNet object categories). For other models, the output could have different meanings. Please consult the actual models specification.

We can register any Keras models that work on images as follows.

```python
from keras.applications import InceptionV3
from sparkdl.udf.keras_image_model import registerKerasImageUDF

from keras.applications import InceptionV3
registerKerasImageUDF("my_keras_inception_udf", InceptionV3(weights="imagenet"))
```

To use a customized Keras model, we can save it and pass the file path as parameter.

```python
# Assume we have a compiled and trained Keras model
model.save('path/to/my/model.h5')
registerKerasImageUDF("my_custom_keras_model_udf", "path/to/my/model.h5")
```

Once the UDF is registered as described above, it can be used in a SQL query.

```sql
SELECT my_custom_keras_model_udf(image) as predictions from my_spark_image_table
```

If there are further preprocessing steps required to prepare the images, the user has the option to provide a preprocessing function `preprocessor`. The `preprocessor` converts a file path into a image array. This function is usually introduced in Keras workflow, as in the following example.

```python
from keras.applications import InceptionV3
from sparkdl.udf.keras_image_model import registerKerasImageUDF

def keras_load_img(fpath):
    from keras.preprocessing.image import load_img, img_to_array
    import numpy as np
    img = load_img(fpath, target_size=(299, 299))
    return img_to_array(img).astype(np.uint8)

registerKerasImageUDF("my_keras_inception_udf", InceptionV3(weights="imagenet"), keras_load_img)

```

## Releases:
- 0.1.0 Alpha release
- 0.2.0 release:
    1. KerasImageFileEstimator API (train a Keras model on image files)
    2. SQL UDF support for Keras models
    3. Added Xception, Resnet50 models to DeepImageFeaturizer/DeepImagePredictor.
