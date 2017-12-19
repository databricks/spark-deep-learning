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
package org.tensorframes.impl

import org.scalatest.FunSuite

import com.databricks.sparkdl.TestSparkContext
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.apache.spark.sql.{functions => sqlfn}

import org.tensorflow.Tensor
import org.tensorframes.{Logging, Shape, ShapeDescription}
import org.tensorframes.dsl.Implicits._
import org.tensorframes.dsl._
import org.tensorframes.{dsl => tf}
import org.apache.spark.sql.sparkdl_stubs._

// Classes used for creating Dataset
// With `import spark.implicits_` we have the encoders
object SqlOpsSchema {
  case class InputCol(a: Double)
  case class DFRow(idx: Long, input: InputCol)
}

class SqlOpsSpec extends FunSuite with TestSparkContext with GraphScoping with Logging {
  lazy val sql = sqlContext  
  import SqlOpsSchema._

  import TestUtils._
  import Shape.Unknown

  test("Must be able to register TensorFlow Graph UDF") {
    val p1 = tf.placeholder[Double](1) named "p1"
    val p2 = tf.placeholder[Double](1) named "p2"
    val a = p1 + p2 named "a"
    val g = buildGraph(a)
    val shapeHints = ShapeDescription(
      Map("p1" -> Shape(1), "p2" -> Shape(1)),
      Seq("p1", "p2"),
      Map("a" -> "a"))

    val udfName = "tfs_test_simple_add"
    val udf = SqlOps.makeUDF(udfName, g, shapeHints, false, false)
    UDFUtils.registerUDF(spark.sqlContext, udfName, udf) // generic UDF registeration
    assert(spark.catalog.functionExists(udfName))
  }

  test("Registered tf.Graph UDF and use in SQL") {
    import spark.implicits._

    val a = tf.placeholder[Double](Unknown) named "inputA"
    val z = a + 2.0 named "z"
    val g = buildGraph(z)

    val shapeHints = ShapeDescription(
      Map("z" -> Shape(1)),
      Seq("z"),
      Map("inputA" -> "a"))

    logDebug(s"graph ${g.toString}")

    // Build the UDF and register
    val udfName = "tfs_test_simple_add"
    val udf = SqlOps.makeUDF(udfName, g, shapeHints, false, false)    
    UDFUtils.registerUDF(spark.sqlContext, udfName, udf) // generic UDF registeration

    // Create a DataFrame
    val inputs = (1 to 100).map(_.toDouble)

    val dfIn = inputs.zipWithIndex.map { case (v, idx) =>
      new DFRow(idx.toLong, new InputCol(v))
    }.toDS.toDF
    dfIn.printSchema()   
    dfIn.createOrReplaceTempView("temp_input_df")

    // Create the query
    val sqlQuery = s"select ${udfName}(input) as output from temp_input_df"
    logDebug(sqlQuery)
    val dfOut = spark.sql(sqlQuery)
    dfOut.printSchema()

    // The UDF maps from StructType => StructType
    // Thus when iterating over the result, each record is a Row of Row    
    val res = dfOut.select("output").collect().map {
      case rowOut @ Row(rowIn @ Row(t)) => 
        //println(rowOut, rowIn, t)
        t.asInstanceOf[Seq[Double]].head
    }

    // Check that all the results are correct
    (res zip inputs).foreach { case (v, u) =>
      assert(v === u + 2.0)
    }
  }

}
