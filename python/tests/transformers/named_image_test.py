# Copyright 2017 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import tensorflow as tf

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StructType, StructField

from sparkdl.image import imageIO
import sparkdl.transformers.keras_applications as keras_apps
from sparkdl.transformers.named_image import (DeepImagePredictor, DeepImageFeaturizer,
                                              _buildTFGraphForName)
from ..tests import SparkDLTestCase
from .image_utils import getSampleImageDF, getSampleImageList


class NamedImageTransformerBaseTestCase(SparkDLTestCase):

    __test__ = False
    name = None

    @classmethod
    def setUpClass(cls):
        super(NamedImageTransformerBaseTestCase, cls).setUpClass()

        imgFiles, images = getSampleImageList()
        imageArray = np.empty((len(images), 299, 299, 3), 'uint8')
        for i, img in enumerate(images):
            assert img is not None and img.mode == "RGB"
            imageArray[i] = np.array(img.resize((299, 299)))
        cls.imageArray = imageArray

        # Predict the class probabilities for the images in our test library using keras API
        # and cache for use by multiple tests.
        cls.appModel = keras_apps.getKerasApplicationModel(cls.name)
        preppedImage = cls.appModel.preprocess(imageArray.astype('float32'))
        kerasPredict = cls.appModel.testKerasModel().predict(preppedImage)
        cls.kerasPredict = kerasPredict

        cls.imageDF = getSampleImageDF().limit(5)


    def test_buildtfgraphforname(self):
        """"
        Run the graph produced by _buildtfgraphforname using tensorflow and compare to keras result.
        """
        imageArray = self.imageArray
        kerasPredict = self.kerasPredict
        modelGraphInfo = _buildTFGraphForName(self.name, False)
        graph = modelGraphInfo["graph"]
        sess = tf.Session(graph=graph)
        with sess.as_default():
            inputTensor = graph.get_tensor_by_name(modelGraphInfo["inputTensorName"])
            outputTensor = graph.get_tensor_by_name(modelGraphInfo["outputTensorName"])
            tfPredict = sess.run(outputTensor, {inputTensor: imageArray})

        self.assertEqual(kerasPredict.shape, tfPredict.shape)
        np.testing.assert_array_almost_equal(kerasPredict, tfPredict)

    def test_DeepImagePredictorNoReshape(self):
        """
        Run sparkDL transformer on manually-resized images and compare result to the
        keras result.
        """
        imageArray = self.imageArray
        kerasPredict = self.kerasPredict
        def rowWithImage(img):
            # return [imageIO.imageArrayToStruct(img.astype('uint8'), imageType.sparkMode)]
            row = imageIO.imageArrayToStruct(img.astype('uint8'), imageIO.SparkMode.RGB)
            # re-order row to avoid pyspark bug
            return [[getattr(row, field.name) for field in imageIO.imageSchema]]

        # test: predictor vs keras on resized images
        rdd = self.sc.parallelize([rowWithImage(img) for img in imageArray])
        dfType = StructType([StructField("image", imageIO.imageSchema)])
        imageDf = rdd.toDF(dfType)

        transformer = DeepImagePredictor(inputCol='image', modelName=self.name,
                                         outputCol="prediction")
        dfPredict = transformer.transform(imageDf).collect()
        dfPredict = np.array([i.prediction for i in dfPredict])

        self.assertEqual(kerasPredict.shape, dfPredict.shape)
        np.testing.assert_array_almost_equal(kerasPredict, dfPredict)

    def test_DeepImagePredictor(self):
        """
        Run sparkDL transformer on raw (original size) images and compare result to
        above keras (using keras resizing) result.
        """
        kerasPredict = self.kerasPredict
        transformer = DeepImagePredictor(inputCol='image', modelName=self.name,
                                         outputCol="prediction",)
        fullPredict = transformer.transform(self.imageDF).collect()
        fullPredict = np.array([i.prediction for i in fullPredict])

        self.assertEqual(kerasPredict.shape, fullPredict.shape)
        # We use a large tolerance below because of differences in the resize step
        # TODO: match keras resize step to get closer prediction
        np.testing.assert_array_almost_equal(kerasPredict, fullPredict, decimal=6)

    def test_prediction_decoded(self):
        output_col = "prediction"
        topK = 10
        transformer = DeepImagePredictor(inputCol="image", outputCol=output_col,
                                         modelName=self.name, decodePredictions=True, topK=topK)
        transformed_df = transformer.transform(self.imageDF)

        collected = transformed_df.collect()
        for row in collected:
            predictions = row[output_col]
            self.assertEqual(len(predictions), topK)
            # TODO: actually check the value of the output to see if they are reasonable
            # e.g. -- compare to just running with keras.

    def test_featurization(self):
        output_col = "prediction"
        transformer = DeepImageFeaturizer(inputCol="image", outputCol=output_col,
                                          modelName=self.name)
        transformed_df = transformer.transform(self.imageDF)

        collected = transformed_df.collect()
        for row in collected:
            predictions = row[output_col]
            self.assertEqual(len(predictions), self.appModel.numOutputFeatures())
            # TODO: actually check the value of the output to see if they are reasonable
            # e.g. -- compare to just running with keras.

    def test_featurizer_in_pipeline(self):
        """
        Tests that the featurizer fits into an MLlib Pipeline.
        Does not test how good the featurization is for generalization.
        """
        featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features",
                                         modelName=self.name)
        lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
        pipeline = Pipeline(stages=[featurizer, lr])

        # add arbitrary labels to run logistic regression
        # TODO: it's weird that the test fails on some combinations of labels. check why.
        label_udf = udf(lambda x: abs(hash(x)) % 2, IntegerType())
        train_df = self.imageDF.withColumn("label", label_udf(self.imageDF["filePath"]))

        lrModel = pipeline.fit(train_df)
        # see if we at least get the training examples right.
        # with 5 examples and e.g. 131k features (for InceptionV3), it ought to.
        pred_df_collected = lrModel.transform(train_df).collect()
        for row in pred_df_collected:
            self.assertEqual(int(row.prediction), row.label)


class NamedImageTransformerInceptionV3Test(NamedImageTransformerBaseTestCase):

    __test__ = True
    name = "InceptionV3"
