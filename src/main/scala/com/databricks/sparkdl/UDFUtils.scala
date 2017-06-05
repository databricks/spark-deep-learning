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

import java.util.ArrayList
import scala.collection.JavaConverters._

import org.apache.spark.internal.Logging
import org.apache.spark.sql.{Column, Row, SparkSession, SQLContext}
import org.apache.spark.sql.catalyst.expressions.{Expression, ScalaUDF}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.DataType

object UDFUtils extends Logging {
  def registerUDF(spark: SparkSession, name: String, udf: UserDefinedFunction): UserDefinedFunction = {
    def builder(children: Seq[Expression]) = udf.apply(children.map(cx => new Column(cx)) : _*).expr
    spark.sessionState.functionRegistry.registerFunction(name, builder)
    udf
  }

  /**
    * Composite a set of UDFs as a single UDF
    * @param ctx an active SQLContext
    * @param name the name of the UDF
    * @param judfs a seuqnce of UDFs (in Java's ArrayList)
    * @return the registered UDF
    */
  def registerCompositeUDF(
    ctx: SQLContext,
    name: String,
    judfs: ArrayList[UserDefinedFunction]): UserDefinedFunction = {

    val udfs = judfs.asScala
    val udf = PipelinedUDF(name, udfs.head, udfs.tail: _*)
    logWarning(s"Registering composite udf $name -> $udf to session ${ctx.sparkSession}")
    registerUDF(ctx.sparkSession, name, udf)
  }

  /**
    * Register a UserDefinedfunction (UDF) as a composition of several UDFs.
    * The UDFs must have already been registered
    * @param ctx an active SQLContext
    * @param name the name of the UDF
    * @param orderedUdfNames a sequence of UDF names in the composition order
    */
  def pipeline(ctx: SQLContext, name: String, orderedUdfNames: Seq[String]) = {
    val registry = ctx.sessionState.functionRegistry
    val builders = orderedUdfNames.flatMap { fname => registry.lookupFunctionBuilder(fname) }
    require(builders.size == orderedUdfNames.size,
      s"all UDFs must have been registered to the SQLContext: $ctx")
    def composedBuilder(children: Seq[Expression]): Expression = {
      builders.foldLeft(children) { case (exprs, fb) => Seq(fb(exprs)) }.head
    }
    registry.registerFunction(name, composedBuilder)
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
    require(exprs.size == 1, exprs)
    val f = fun(exprs.head)
    val inner = exprs.toSeq.map(_.toString()).mkString(", ")
    val name = s"$opName($inner)"
    new Column(ScalaUDF(f, dataType, exprs.map(_.expr), Nil)).alias(name)
  }
}
