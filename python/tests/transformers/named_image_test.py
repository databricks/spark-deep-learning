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

from keras.applications import resnet50
import numpy as np
import os
from PIL import Image
from scipy import spatial
import tensorflow as tf

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType, StructType, StructField

from sparkdl.image import imageIO
import sparkdl.transformers.keras_applications as keras_apps
from sparkdl.transformers.named_image import (DeepImagePredictor, DeepImageFeaturizer,
                                              _buildTFGraphForName)

from pyspark.ml.image import ImageSchema

from ..tests import SparkDLTestCase, SparkDLTempDirTestCase
from .image_utils import getSampleImageDF
from.image_utils import getImageFiles


class KerasApplicationModelTestCase(SparkDLTestCase):
    def test_getKerasApplicationModelError(self):
        self.assertRaises(ValueError, keras_apps.getKerasApplicationModel, "NotAModelABC")

    def test_imagenet_preprocess_input(self):
        # compare our tf implementation to the np implementation in keras
        image = np.zeros((256, 256, 3))

        sess = tf.Session()
        with sess.as_default():
            x = tf.placeholder(tf.float32, shape=[256, 256, 3])
            processed = keras_apps._imagenet_preprocess_input(x, (256, 256)),
            sparkdl_preprocessed_input = sess.run(processed, {x: image})

        keras_preprocessed_input = resnet50.preprocess_input(np.expand_dims(image, axis=0))

        # NOTE: precision errors occur for decimal > 5
        np.testing.assert_array_almost_equal(sparkdl_preprocessed_input, keras_preprocessed_input,
                                             decimal=5)


class NamedImageTransformerBaseTestCase(SparkDLTestCase):
    """
    The tests here are written for Keras application -based models but test the
    NamedImageTransformer API. If we add non-Keras application -based models we
    will want to refactor.
    """

    __test__ = False
    name = None
    # Allow subclasses to force number of partitions - a hack to avoid OOM issues
    numPartitionsOverride = None
    featurizerCompareDigitsExact = 5
    featurizerCompareDigitsCosine = 1

    @classmethod
    def getSampleImageList(cls):
        shape = cls.appModel.inputShape()
        imageFiles = getImageFiles()
        images = [imageIO.PIL_to_imageStruct(Image.open(f).resize(shape)) for f in imageFiles]
        return imageFiles, np.array(images)

    @classmethod
    def setUpClass(cls):
        super(NamedImageTransformerBaseTestCase, cls).setUpClass()
        cls.appModel = keras_apps.getKerasApplicationModel(cls.name)
        imgFiles, imageArray = cls.getSampleImageList()
        cls.imageArray = imageArray
        cls.imgFiles = imgFiles
        cls.fileOrder = {imgFiles[i].split('/')[-1]: i for i in range(len(imgFiles))}
        # Predict the class probabilities for the images in our test library using keras API
        # and cache for use by multiple tests.
        preppedImage = cls.appModel._testPreprocess(imageArray.astype('float32'))
        cls.preppedImage = preppedImage
        cls.kerasPredict = cls.appModel._testKerasModel(
            include_top=True).predict(preppedImage, batch_size=1)
        cls.kerasFeatures = cls.appModel._testKerasModel(include_top=False).predict(preppedImage)

        cls.imageDF = getSampleImageDF().limit(5)
        if(cls.numPartitionsOverride):
            cls.imageDf = cls.imageDF.coalesce(cls.numPartitionsOverride)

    def _sortByFileOrder(self, ary):
        """
        This is to ensure we are comparing compatible sequences of predictions.
        Sorts the results according to the order in which the files have been read by python.
        Note: Java and python can read files in different order.
        """
        fileOrder = self.fileOrder
        return sorted(ary, key=lambda x: fileOrder[x['image']['origin'].split('/')[-1]])

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
        np.testing.assert_array_almost_equal(kerasPredict,
                                             tfPredict,
                                             decimal=self.featurizerCompareDigitsExact)

    def _rowWithImage(self, img):
        row = imageIO.imageArrayToStruct(img.astype('uint8'))
        # re-order row to avoid pyspark bug
        return [[getattr(row, field.name)
                 for field in ImageSchema.imageSchema['image'].dataType]]

    def test_DeepImagePredictorNoReshape(self):
        """
        Run sparkDL predictor on manually-resized images and compare result to the
        keras result.
        """
        imageArray = self.imageArray
        kerasPredict = self.kerasPredict

        # test: predictor vs keras on resized images
        rdd = self.sc.parallelize([self._rowWithImage(img) for img in imageArray])
        dfType = ImageSchema.imageSchema
        imageDf = rdd.toDF(dfType)
        if self.numPartitionsOverride:
            imageDf = imageDf.coalesce(self.numPartitionsOverride)

        transformer = DeepImagePredictor(inputCol='image', modelName=self.name,
                                         outputCol="prediction")
        dfPredict = transformer.transform(imageDf).collect()
        dfPredict = np.array([i.prediction for i in dfPredict])

        self.assertEqual(kerasPredict.shape, dfPredict.shape)
        np.testing.assert_array_almost_equal(kerasPredict,
                                             dfPredict,
                                             decimal=self.featurizerCompareDigitsExact)

    def test_DeepImagePredictor(self):
        """
        Tests that predictor returns (almost) the same values as Keras.
        """
        kerasPredict = self.kerasPredict
        transformer = DeepImagePredictor(inputCol='image', modelName=self.name,
                                         outputCol="prediction",)
        fullPredict = self._sortByFileOrder(transformer.transform(self.imageDF).collect())
        fullPredict = np.array([i.prediction for i in fullPredict])
        self.assertEqual(kerasPredict.shape, fullPredict.shape)
        np.testing.assert_array_almost_equal(kerasPredict,
                                             fullPredict,
                                             decimal=self.featurizerCompareDigitsExact)

    def test_prediction_decoded(self):
        """
        Tests that predictor with decoded=true returns reasonable values.
        """
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

    def test_featurization_no_reshape(self):
        """
        Run sparkDL predictor on manually-resized images and compare result to the
        keras result.
        """
        imageArray = self.imageArray
        # test: predictor vs keras on resized images
        rdd = self.sc.parallelize([self._rowWithImage(img) for img in imageArray])
        dfType = ImageSchema.imageSchema
        imageDf = rdd.toDF(dfType)
        if self.numPartitionsOverride:
            imageDf = imageDf.coalesce(self.numPartitionsOverride)
        transformer = DeepImageFeaturizer(inputCol='image', modelName=self.name,
                                          outputCol="features")
        dfFeatures = transformer.transform(imageDf).collect()
        dfFeatures = np.array([i.features for i in dfFeatures])
        kerasReshaped = self.kerasFeatures.reshape(self.kerasFeatures.shape[0], -1)
        np.testing.assert_array_almost_equal(kerasReshaped,
                                             dfFeatures,
                                             decimal=self.featurizerCompareDigitsExact)


    def test_featurization(self):
        """
        Tests that featurizer returns (almost) the same values as Keras.
        """
        # Since we use different libraries for image resizing (PIL in python vs. java.awt.Image in scala),
        # the result will not match keras exactly. In fact the best we can do is a "somewhat similar" result.
        # At least compare cosine distance is < 1e-2
        featurizer_sc = DeepImageFeaturizer(modelName=self.name, inputCol="image",
                                            outputCol="features", scaleHint="SCALE_FAST")
        features_sc = np.array([i.features for i in featurizer_sc.transform(
            self.imageDF).select("features").collect()])
        kerasReshaped = self.kerasFeatures.reshape(self.kerasFeatures.shape[0], -1)
        diffs = [
            spatial.distance.cosine(
                kerasReshaped[i],
                features_sc[i]) for i in range(
                len(features_sc))]
        np.testing.assert_array_almost_equal(0, diffs, decimal=self.featurizerCompareDigitsCosine)

    def test_featurizer_in_pipeline(self):
        """
        Tests that featurizer fits into an MLlib Pipeline.
        Does not test how good the featurization is for generalization.
        """
        featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features",
                                         modelName=self.name)
        lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
        pipeline = Pipeline(stages=[featurizer, lr])

        # add arbitrary labels to run logistic regression
        # TODO: it's weird that the test fails on some combinations of labels. check why.
        label_udf = udf(lambda x: abs(hash(x)) % 2, IntegerType())
        train_df = self.imageDF.withColumn("label", label_udf(self.imageDF["image"]["origin"]))

        lrModel = pipeline.fit(train_df)
        # see if we at least get the training examples right.
        # with 5 examples and e.g. 131k features (for InceptionV3), it ought to.
        pred_df_collected = lrModel.transform(train_df).collect()
        for row in pred_df_collected:
            self.assertEqual(int(row.prediction), row.label)


class DeepImageFeaturizerPersistenceTest(SparkDLTempDirTestCase):
    def test_inception(self):
        transformer0 = DeepImageFeaturizer(inputCol='image', modelName="InceptionV3",
                                           outputCol="features0", scaleHint="SCALE_FAST")
        dst_path = os.path.join(self.tempdir, "featurizer")
        transformer0.save(dst_path)
        transformer1 = DeepImageFeaturizer.load(dst_path)
        self.assertEqual(transformer0.uid, transformer1.uid)
        self.assertEqual(type(transformer0.uid), type(transformer1.uid))
        for x in transformer0._paramMap.keys():
            self.assertEqual(transformer1.uid, x.parent,
                             "Loaded DeepImageFeaturizer instance uid (%s) did not match Param's uid (%s)"
                             % (transformer1.uid, transformer1.scaleHint.parent))
        self.assertEqual(transformer0._paramMap, transformer1._paramMap,
                         "Loaded DeepImageFeaturizer instance params (%s) did not match "
                         % str(transformer1._paramMap) +
                         "original values (%s)" % str(transformer0._paramMap))
        self.assertEqual(transformer0._defaultParamMap, transformer1._defaultParamMap,
                         "Loaded DeepImageFeaturizer instance default params (%s) did not match "
                         % str(transformer1._defaultParamMap) +
                         "original defaults (%s)" % str(transformer0._defaultParamMap))
