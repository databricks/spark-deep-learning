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

import org.scalatest.FunSuite

import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{StructField, StructType}


class DeepImageFeaturizerSuite extends FunSuite with TestSparkContext with DefaultReadWriteTest {

  var data: DataFrame = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    val imageDir = getClass.getResource("/sparkdl/test-image-collection").getFile
    data = ImageSchema.readImages(imageDir)
  }

  test ("Test named image featurizer runs on runs on image dataframes.") {
    val myData = data.withColumn("myInput", col("image"))
    val outputColName = "myOutput"
    val featurizer = new DeepImageFeaturizer()
      .setModelName("_test")
      .setInputCol("myInput")
      .setOutputCol(outputColName)
    val transformed = featurizer.transform(myData)

    assert(transformed.columns contains outputColName, "The expected output column was not created.")
    assert(featurizer.transformSchema(myData.schema) === transformed.schema)

    // check that we can materialize a row, and the type is Vector.
    val result = transformed.select(col(outputColName)).collect()
    assert(result.forall { r: Row => r.getAs[Vector](0).size == 24 })
  }

  test ("Test schema validation.") {
    val missingInputColumn = "missingInputColumn"
    val outputColumn = "outputColumn"
    val featurizer = new DeepImageFeaturizer()
      .setInputCol(missingInputColumn)
      .setOutputCol(outputColumn)

    import spark.implicits._
    val data = sqlContext.createDataset(0 until 100).toDF("columnName")

    assertThrows[IllegalArgumentException] {
      featurizer.transformSchema(data.schema)
    }

    assertThrows[IllegalArgumentException] {
      val hasColumnWithWrongType = data.withColumn(missingInputColumn, lit("str"))
      featurizer.transformSchema(hasColumnWithWrongType.schema)
    }

    assertThrows[IllegalArgumentException] {
      val hasOutputColumn = data.withColumn(outputColumn, lit("str"))
      featurizer.transformSchema(hasOutputColumn.schema)
    }
  }

  test("Test test_net on a known data sample.") {
    import ImageUtilsSuite.biggerImage
    import ImageUtilsSuite.smallerImage

    val outputColName = "myOutput"
    val featurizer = new DeepImageFeaturizer()
      .setModelName("_test")
      .setInputCol("myInput")
      .setOutputCol(outputColName)

    val dfSchema = StructType(Array(StructField("myInput", ImageSchema.columnSchema, false)))
    val rdd = sc.parallelize(Seq(
      Row(biggerImage),
      Row(smallerImage))
    )
    val knownData = sqlContext.createDataFrame(rdd, dfSchema)

    val features = featurizer.transform(knownData)
    val expectedFeatures = Vectors.dense(59, 43, 53, 72, 43, 30, 42, 75, 53, 19, 26, 85, 81, 63,
      66, 113, 76, 49, 56, 63, 97, 89, 84, 53)
    val vector = features.select(col(outputColName)).collect.foreach{ row =>
      val vector = row.getAs[Vector](0)
      assert(vector === expectedFeatures,
        "DeepImageFeaturizer, using test_net, featurizer did not produce the output we expect " +
          "to see."
      )
    }
  }

  test("DeepImageFeaturizer modelName param throws if invalid or no model name is provided.") {
    val featurizer = new DeepImageFeaturizer()
      // Do not set model name
      .setInputCol("image")
      .setOutputCol("someOutput")

    assertThrows[NoSuchElementException] {
      featurizer.transform(data)
    }

    assertThrows[IllegalArgumentException] {
      featurizer.setModelName("noSuchModel")
    }
  }

  test("DeepImageFeaturizer persistence") {
    val featurizer = new DeepImageFeaturizer()
      .setModelName("_test")
      .setInputCol("myInput")
      .setOutputCol("myOutput")
    testDefaultReadWrite(featurizer)
  }

  test("DeepImageFeaturizer accepts nullable") {
    val nullableImageSchema = StructType(
      data.schema("image").dataType.asInstanceOf[StructType]
        .fields.map(_.copy(nullable = true)))
    val nullableSchema = StructType(StructField("image", nullableImageSchema, true) :: Nil)
    val featurizer = new DeepImageFeaturizer()
      .setModelName("_test")
      .setInputCol("image")
      .setOutputCol("features")
    withClue("featurizer should accept nullable schemas") {
      featurizer.transformSchema(nullableSchema)
    }
  }
}
