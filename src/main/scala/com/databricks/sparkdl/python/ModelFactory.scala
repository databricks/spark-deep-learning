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
package com.databricks.sparkdl.python

import java.nio.file.{Files, Paths}
import java.util

import org.apache.log4j.PropertyConfigurator

import scala.collection.JavaConverters._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.sparkdl_stubs.UDFUtils
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.tensorflow.framework.GraphDef
import org.tensorframes.{Shape, ShapeDescription}
import org.tensorframes.impl.{SerializedGraph, SqlOps, TensorFlowOps}

import com.databricks.sparkdl.Logging

/**
  *
  */
// TODO: merge with the python factory eventually, this is essentially copy/paste
class GraphModelFactory() extends Logging {
  private var _shapeHints: ShapeDescription = ShapeDescription.empty
  // TODO: this object may leak because of Py4J -> do not hold to large objects here.
  private var _graph: SerializedGraph = null
  private var _graphPath: Option[String] = None
  private var _sqlCtx: SQLContext = null

  def initialize_logging(): Unit = initialize_logging("org/tensorframes/log4j.properties")

  /**
   * Performs some logging initialization before spark has the time to do it.
   *
   * Because of the the current implementation of PySpark, Spark thinks it runs as an interactive
   * console and makes some mistake when setting up log4j.
   */
  private def initialize_logging(file: String): Unit = {
    Option(this.getClass.getClassLoader.getResource(file)) match {
      case Some(url) =>
        PropertyConfigurator.configure(url)
      case None =>
        System.err.println(s"$this Could not load logging file $file")
    }
  }

  def shape(
      shapeHintsNames: util.ArrayList[String],
      shapeHintShapes: util.ArrayList[util.ArrayList[Int]]): this.type = {
    val s = shapeHintShapes.asScala.map(_.asScala.toSeq).map(x => Shape(x: _*))
    _shapeHints = _shapeHints.copy(out = shapeHintsNames.asScala.zip(s).toMap)
    this
  }

  def fetches(fetchNames: util.ArrayList[String]): this.type = {
    _shapeHints = _shapeHints.copy(requestedFetches = fetchNames.asScala)
    this
  }

  def graph(bytes: Array[Byte]): this.type = {
    _graph = SerializedGraph.create(bytes)
    this
  }

  def graphFromFile(filename: String): this.type = {
    _graphPath = Option(filename)
    this
  }

  def sqlContext(ctx: SQLContext): this.type = {
    _sqlCtx = ctx
    this
  }

  def inputs(
      placeholderPaths: util.ArrayList[String],
      fieldNames: util.ArrayList[String]): this.type = {
    require(placeholderPaths.size() == fieldNames.size(), (placeholderPaths.asScala, fieldNames.asScala))
    val map = placeholderPaths.asScala.zip(fieldNames.asScala).toMap
    _shapeHints = _shapeHints.copy(inputs = map)
    this
  }

  /**
    * Builds a java UDF based on the following input.
    */
  def makeUDF(udfName: String, applyBlocks: Boolean): UserDefinedFunction = {
    SqlOps.makeUDF(udfName, buildGraphDef(), _shapeHints,
      applyBlocks = applyBlocks, flattenStruct = true)
  }

  /**
    * Builds a java UDF based on the following input.
    */
  def makeUDF(udfName: String, applyBlocks: Boolean, flattenStruct: Boolean): UserDefinedFunction = {
    SqlOps.makeUDF(udfName, buildGraphDef(), _shapeHints,
      applyBlocks = applyBlocks, flattenStruct = flattenStruct)
  }

  /**
   * Registers a TF UDF under the given name in Spark.
   * @param udfName the name of the UDF
   * @param blocked indicates that the UDF should be applied block-wise.
   * @return
   */
  def registerUDF(udfName: String, blocked: java.lang.Boolean): UserDefinedFunction = {
    assert(_sqlCtx != null)
    val udf = makeUDF(udfName, blocked)
    logger.warn(s"Registering udf $udfName -> $udf to session ${_sqlCtx.sparkSession}")
    UDFUtils.registerUDF(_sqlCtx.sparkSession, udfName, udf)
  }

  private def buildGraphDef(): GraphDef = {
    _graphPath match {
      case Some(p) =>
        val path = Paths.get(p)
        val bytes = Files.readAllBytes(path)
        TensorFlowOps.readGraphSerial(SerializedGraph.create(bytes))
      case None =>
        assert(_graph != null)
        TensorFlowOps.readGraphSerial(_graph)
    }
  }

}
