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

import org.apache.spark.internal.Logging
import org.apache.spark.sql.{Column, Row, SQLContext}
import org.apache.spark.sql.catalyst.FunctionIdentifier
import org.apache.spark.sql.catalyst.expressions.{Expression, ScalaUDF}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.DataType

object UDFUtils extends Logging {
  /**
   * Register a UDF to the given SparkSession, so as to expose it in Spark SQL
   * @param spark the SparkSession to which we want to register the UDF
   * @param name registered to the provided SparkSession
   * @param udf the actual body of the UDF
   * @return the registered UDF
   */
  def registerUDF(sqlCtx: SQLContext, name: String, udf: UserDefinedFunction): UserDefinedFunction = {
    def builder(children: Seq[Expression]) = udf.apply(children.map(cx => new Column(cx)) : _*).expr
    val registry = sqlCtx.sessionState.functionRegistry
    registry.registerFunction(FunctionIdentifier(name), builder)
    udf
  }

  /**
   * Register a UserDefinedfunction (UDF) as a composition of several UDFs.
   * The UDFs must have already been registered
   * @param spark the SparkSession to which we want to register the UDF
   * @param name registered to the provided SparkSession
   * @param orderedUdfNames a sequence of UDF names in the composition order
   */
  def registerPipeline(sqlCtx: SQLContext, name: String, orderedUdfNames: Seq[String]) = {
    val registry = sqlCtx.sessionState.functionRegistry
    val builders = orderedUdfNames.flatMap { fname => registry.lookupFunctionBuilder(FunctionIdentifier(fname)) }
    require(builders.size == orderedUdfNames.size,
      s"all UDFs must have been registered to the SQL context: $sqlCtx")
    def composedBuilder(children: Seq[Expression]): Expression = {
      builders.foldLeft(children) { case (exprs, fb) => Seq(fb(exprs)) }.head
    }
    registry.registerFunction(FunctionIdentifier(name), composedBuilder)
  }
}


/**
 * Registering a set of UserDefinedFunctions (UDF)
 */
class PipelinedUDF(
  opName: String,
  udfs: Seq[UserDefinedFunction],
  returnType: DataType) extends UserDefinedFunction(null, returnType, None) {
  require(udfs.nonEmpty)

  override def apply(exprs: Column*): Column = {
    val start = udfs.head.apply(exprs: _*)
    var rest = start
    for (udf <- udfs.tail) {
      rest = udf.apply(rest)
    }
    val inner = exprs.toSeq.map(_.toString()).mkString(", ")
    val name = s"$opName($inner)"
    rest.alias(name)
  }
}

object PipelinedUDF {
  def apply(opName: String, fn: UserDefinedFunction, fns: UserDefinedFunction*): UserDefinedFunction = {
    if (fns.isEmpty) return fn
    new PipelinedUDF(opName, Seq(fn) ++ fns, fns.last.dataType)
  }
}


class RowUDF(
  opName: String,
  fun: Column => (Any => Row),
  returnType: DataType) extends UserDefinedFunction(null, returnType, None) {

  override def apply(exprs: Column*): Column = {
    require(exprs.size == 1, "only support one function")
    val f = fun(exprs.head)
    val inner = exprs.toSeq.map(_.toString()).mkString(", ")
    val name = s"$opName($inner)"
    new Column(ScalaUDF(f, dataType, exprs.map(_.expr), Nil)).alias(name)
  }
}
