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
package org.apache.spark.sql.sparkdl_stubs

import org.scalatest.FunSuite

import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._

import com.databricks.sparkdl.TestSparkContext

/**
 * Testing UDF registration
 */
class SparkDLStubSuite extends FunSuite with TestSparkContext {

  test("Registered UDF must be found") {
    val udfName = "sparkdl_test_udf"
    val udfImpl = { (x: Int, y: Int) => x + y }
    UDFUtils.registerUDF(spark.sqlContext, udfName, udf(udfImpl))
    assert(spark.catalog.functionExists(udfName))
  }

  test("Registered piped UDF must be found") {
    val udfName = "sparkdl_test_piped_udf"

    UDFUtils.registerUDF(spark.sqlContext, s"${udfName}_0",
      udf({ (x: Int, y: Int) => x + y}))
    UDFUtils.registerUDF(spark.sqlContext, s"${udfName}_1",
      udf({ (z: Int) => z * 2}))
    UDFUtils.registerUDF(spark.sqlContext, s"${udfName}_2",
      udf({ (w: Int) => w * w + 3}))
    
    UDFUtils.registerPipeline(spark.sqlContext, udfName,
      (0 to 2).map { idx => s"${udfName}_$idx" })

    assert(spark.catalog.functionExists(udfName))
  }

  test("Using piped UDF in SQL") {
    val udfName = "sparkdl_test_piped_udf"

    UDFUtils.registerUDF(spark.sqlContext, s"${udfName}_add",
      udf({ (x: Int, y: Int) => x + y}))
    UDFUtils.registerUDF(spark.sqlContext, s"${udfName}_mul",
      udf({ (z: Int) => z * 2}))
    
    UDFUtils.registerPipeline(spark.sqlContext, udfName, Seq(s"${udfName}_add", s"${udfName}_mul"))

    import spark.implicits._
    val df = Seq(1 -> 1, 2 -> 2).toDF("x", "y")
    df.createOrReplaceTempView("piped_udf_input_df")
    df.printSchema()

    val sqlQuery = s"select x, y, $udfName(x, y) as res from piped_udf_input_df"
    println(sqlQuery)
    val dfRes = spark.sql(sqlQuery)
    dfRes.printSchema()
    dfRes.collect().map { case Row(x: Int, y: Int, res: Int) =>      
      assert((x + y) * 2 === res)
    }
  }

}
