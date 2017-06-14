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

import scala.reflect.runtime.universe._

import org.scalatest.{FunSuite, BeforeAndAfterAll}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, DataFrame, SQLContext, SparkSession}

// This context is used for all tests in this project
trait TestSparkContext extends BeforeAndAfterAll { self: FunSuite =>
  @transient var sc: SparkContext = _
  @transient var sqlContext: SQLContext = _
  @transient lazy val spark: SparkSession = {
    val conf = new SparkConf()
      .setMaster("local[*]")      
      .setAppName("Spark-Deep-Learning-Test")
      .set("spark.ui.port", "4079")
      .set("spark.sql.shuffle.partitions", "4")  // makes small tests much faster

    SparkSession.builder().config(conf).getOrCreate()
  }

  override def beforeAll() {
    super.beforeAll()
    sc = spark.sparkContext
    sqlContext = spark.sqlContext
    import spark.implicits._
  }

  override def afterAll() {
    sqlContext = null
    if (sc != null) {
      sc.stop()
    }
    sc = null
    super.afterAll()
  }

  def makeDF[T: TypeTag](xs: Seq[T], col: String): DataFrame = {
    sqlContext.createDataFrame(xs.map(Tuple1.apply)).toDF(col)
  }

  def compareRows(r1: Array[Row], r2: Seq[Row]): Unit = {
    val a = r1.sortBy(_.toString())
    val b = r2.sortBy(_.toString())
    assert(a === b)
  }
}
