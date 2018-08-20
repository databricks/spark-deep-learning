#
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

from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras.applications import InceptionV3
from keras.applications import inception_v3 as iv3
import keras.backend as K
from keras.layers import Activation, Dense, Flatten, Input
from keras.models import Sequential

from pyspark import SparkContext
from pyspark.sql import DataFrame, Row
from pyspark.sql.functions import udf
from pyspark.ml.image import ImageSchema

from sparkdl.graph.builder import IsolatedSession
from sparkdl.graph.tensorframes_udf import makeGraphUDF
import sparkdl.graph.utils as tfx
from sparkdl.udf.keras_image_model import registerKerasImageUDF
from sparkdl.utils import jvmapi as JVMAPI
from sparkdl.image.imageIO import imageArrayToStruct
from sparkdl.image.imageIO import _reverseChannels
from ..tests import SparkDLTestCase
from ..transformers.image_utils import getSampleImagePathsDF


def get_image_paths_df(sqlCtx):
    df = getSampleImagePathsDF(sqlCtx, "fpath")
    df.createOrReplaceTempView("_test_image_paths_df")
    return df


class SqlUserDefinedFunctionTest(SparkDLTestCase):

    def _assert_function_exists(self, fh_name):
        spark_fh_name_set = set([fh.name for fh in self.session.catalog.listFunctions()])
        self.assertTrue(fh_name in spark_fh_name_set)

    def test_simple_keras_udf(self):
        """ Simple Keras sequential model """
        # Notice that the input layer for a image UDF model
        # must be of shape (width, height, numChannels)
        # The leading batch size is taken care of by Keras
        with IsolatedSession(using_keras=True) as issn:
            model = Sequential()
            # Make the test model simpler to increase the stability of travis tests
            model.add(Flatten(input_shape=(640, 480, 3)))
            # model.add(Dense(64, activation='relu'))
            model.add(Dense(16, activation='softmax'))
            # Initialize the variables
            init_op = tf.global_variables_initializer()
            issn.run(init_op)
            makeGraphUDF(issn.graph,
                         'my_keras_model_udf',
                         model.outputs,
                         {tfx.op_name(model.inputs[0], issn.graph): 'image_col'})
            # Run the training procedure
            # Export the graph in this IsolatedSession as a GraphFunction
            # gfn = issn.asGraphFunction(model.inputs, model.outputs)
            fh_name = "test_keras_simple_sequential_model"
            registerKerasImageUDF(fh_name, model)

        self._assert_function_exists(fh_name)

    def test_pretrained_keras_udf(self):
        """ Must be able to register a pretrained image model as UDF """
        # Register an InceptionV3 model
        fh_name = "test_keras_pretrained_iv3_model"
        registerKerasImageUDF(fh_name,
                              InceptionV3(weights="imagenet"))
        self._assert_function_exists(fh_name)

    def test_composite_udf(self):
        """ Composite Keras Image UDF registration """
        df = get_image_paths_df(self.sql)

        def keras_load_img(fpath):
            from keras.preprocessing.image import load_img, img_to_array
            import numpy as np
            from pyspark.sql import Row
            img = load_img(fpath, target_size=(299, 299))
            return img_to_array(img).astype(np.uint8)

        def pil_load_spimg(fpath):
            from PIL import Image
            import numpy as np
            img_arr = np.array(Image.open(fpath), dtype=np.uint8)
            # PIL is RGB, image schema is BGR => need to flip the channels
            return imageArrayToStruct(_reverseChannels(img_arr))

        def keras_load_spimg(fpath):
            # Keras loads image in RGB order, ImageSchema expects BGR => need to flip
            return imageArrayToStruct(_reverseChannels(keras_load_img(fpath)))

        # Load image with Keras and store it in our image schema
        JVMAPI.registerUDF('keras_load_spimg', keras_load_spimg,
                           ImageSchema.imageSchema['image'].dataType)
        JVMAPI.registerUDF('pil_load_spimg', pil_load_spimg,
                           ImageSchema.imageSchema['image'].dataType)

        # Register an InceptionV3 model
        registerKerasImageUDF("iv3_img_pred",
                              InceptionV3(weights="imagenet"),
                              keras_load_img)

        run_sql = self.session.sql

        # Choice 1: manually chain the functions in SQL
        df1 = run_sql("select iv3_img_pred(keras_load_spimg(fpath)) as preds from _test_image_paths_df")
        preds1 = np.array(df1.select("preds").rdd.collect())

        # Choice 2: build a pipelined UDF and directly use it in SQL
        JVMAPI.registerPipeline("load_img_then_iv3_pred", ["keras_load_spimg", "iv3_img_pred"])
        df2 = run_sql("select load_img_then_iv3_pred(fpath) as preds from _test_image_paths_df")
        preds2 = np.array(df2.select("preds").rdd.collect())

        # Choice 3: create the image tensor input table first and apply the Keras model
        df_images = run_sql("select pil_load_spimg(fpath) as image from _test_image_paths_df")
        df_images.createOrReplaceTempView("_test_images_df")
        df3 = run_sql("select iv3_img_pred(image) as preds from _test_images_df")
        preds3 = np.array(df3.select("preds").rdd.collect())

        self.assertTrue(len(preds1) == len(preds2))
        np.testing.assert_allclose(preds1, preds2)
        np.testing.assert_allclose(preds2, preds3)

    def test_map_rows_sql_1(self):
        data = [Row(x=float(x)) for x in range(5)]
        df = self.sql.createDataFrame(data)
        with IsolatedSession() as issn:
            # The placeholder that corresponds to column 'x' as a whole column
            x = tf.placeholder(tf.double, shape=[], name="x")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # Let's register these computations in SQL.
            makeGraphUDF(issn.graph, "map_rows_sql_1", [z])

        # Here we go, for the SQL users, straight from PySpark.
        df2 = df.selectExpr("map_rows_sql_1(x) AS z")
        print("df2 = %s" % df2)
        data2 = df2.collect()
        assert data2[0].z == 3.0, data2

    def test_map_blocks_sql_1(self):
        data = [Row(x=float(x)) for x in range(5)]
        df = self.sql.createDataFrame(data)
        with IsolatedSession() as issn:
            # The placeholder that corresponds to column 'x' as a whole column
            x = tf.placeholder(tf.double, shape=[None], name="x")
            # The output that adds 3 to x
            z = tf.add(x, 3, name='z')
            # Let's register these computations in SQL.
            makeGraphUDF(issn.graph, "map_blocks_sql_1", [z], blocked=True)

        # Here we go, for the SQL users, straight from PySpark.
        df2 = df.selectExpr("map_blocks_sql_1(x) AS z")
        print("df2 = %s" % df2)
        data2 = df2.collect()
        assert len(data2) == 5, data2
        assert data2[0].z == 3.0, data2
