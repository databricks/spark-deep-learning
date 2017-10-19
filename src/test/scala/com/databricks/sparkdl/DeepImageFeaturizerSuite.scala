/*
 * Copyright 2017 Databricks, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.databricks.sparkdl

import org.apache.spark.image.ImageSchema
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructField, StructType}
import org.scalatest.FunSuite

class DeepImageFeaturizerSuite extends FunSuite with TestSparkContext {
  test ("Test named image featurizer runs on our image samples"){
    val outputColName = "myOutput"
    val featurizer = new DeepImageFeaturizer()
      .setInputCol("myInput")
      .setOutputCol(outputColName)
    val imageDir = getClass.getResource("/images").getFile
    val data = ImageSchema.readImages(imageDir)
      .withColumn("myInput", col("image"))
    val d1 = featurizer.transform(data)

    assert(d1.columns contains outputColName, "The expected output column was not created.")

    // check that we can materialize a row, and the type is Vector.
    val vect = d1.select(col(outputColName)).first().getAs[Vector](0)

    // Test that we can keep columns with names that match the output graph node name. test_net's
    // output node is named "output".
    val dataWithOutput = data.withColumn("output", lit(3))
    val d2 = featurizer.transform(dataWithOutput)
    assert(d2.columns contains "output")

    assert(featurizer.transformSchema(dataWithOutput.schema).fieldNames === d2.columns)
    assert(featurizer.transformSchema(dataWithOutput.schema) === d2.schema)
  }

  test("Test test_net on a known data sample.") {
    import ImageUtilsSuite.biggerImage
    import ImageUtilsSuite.smallerImage

    val outputColName = "myOutput"
    val featurizer = new DeepImageFeaturizer()
      .setInputCol("myInput")
      .setOutputCol(outputColName)

    val dfSchema = StructType(Array(StructField("myInput", ImageSchema.columnSchema, false)))
    val rdd = sc.parallelize(Seq(
      Row(biggerImage),
      Row(smallerImage))
    )
    val knownData = sqlContext.createDataFrame(rdd, dfSchema)

    val features = featurizer.transform(knownData)
    val vector = features.select(col(outputColName)).first().getAs[Vector](0)
    assert(
      vector === Vectors.dense(59, 43, 53, 72, 43, 30, 42, 75, 53, 19, 26, 85, 81, 63, 66, 113,
        76, 49, 56, 63, 97, 89, 84, 53),
      "test_net featurizer did not produce the output we expect to see."
    )
  }

}
