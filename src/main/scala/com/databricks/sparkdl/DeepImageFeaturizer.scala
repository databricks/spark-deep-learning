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

import scala.collection.JavaConverters._

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{DataTypeShim, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.tensorflow.framework.GraphDef
import org.tensorframes.impl.DebugRowOps
import org.tensorframes.{Shape, ShapeDescription}


class DeepImageFeaturizer(override val uid: String) extends Transformer with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("deepImageFeaturizer"))

  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  final val scaleHint: Param[String] = new Param(this, "scaleHint", "hint which method to use for resizing.",
    (name: String) => DeepImageFeaturizer.scaleHints.contains(name))
  setDefault(scaleHint, "SCALE_AREA_AVERAGING")
  final val modelName: Param[String] = new Param[String](
    this,
    "modelName",
    "name of featurizer model.",
    (name: String) => DeepImageFeaturizer.supportedModelMap.contains(name)
  )

  private val RESIZED_IMAGE_COL = "__sparkdl_imageResized"
  private val INPUT_BUFFER_COL = "__sparkdl_imageBuffer"


  private def validateSchema(schema: StructType): Unit = {
    val inputColumnName = getInputCol
    require(
      schema.fieldNames.contains(inputColumnName),
      s"Input DataFrame must contain column named $inputColumnName")
    require(
      ! schema.fieldNames.contains(getOutputCol),
      s"Input DataFrame cannot already contain a column with name $getOutputCol"
    )
    val fieldIndex = schema.fieldIndex(inputColumnName)
    val colType = schema.fields(fieldIndex).dataType
    require(
      DataTypeShim.equalsIgnoreNullability(colType, ImageSchema.columnSchema),
      s"inputCol must be an image column with schema ImageSchema.columnSchema, got ${colType}"
    )
  }

  override def transformSchema(schema: StructType): StructType = {
    validateSchema(schema)
    schema.add(getOutputCol, VectorType)
  }

  override def copy(extra: ParamMap): Transformer = {
    defaultCopy(extra)
  }

  def getModelName: String = getOrDefault(modelName)

  def getInputCol: String = getOrDefault(inputCol)

  def getOutputCol: String = getOrDefault(outputCol)

  def getScaleHint: String = getOrDefault(scaleHint)

  def setModelName(value: String): this.type = {
    set(modelName, value)
    this
  }

  def setInputCol(value: String): this.type = {
    set(inputCol, value)
    this
  }

  def setOutputCol(value: String): this.type = {
    set(outputCol, value)
    this
  }

  def setScaleHint(value: String): this.type = {
    set(scaleHint, value)
    this
  }

  def transform(dataFrame: Dataset[_]): DataFrame = {
    validateSchema(dataFrame.schema)
    val model = DeepImageFeaturizer.supportedModelMap(getModelName)

    val imSchema = ImageSchema.columnSchema
    val height = model.height
    val width = model.width

    val resizeUdf = udf((image: Row) => ImageUtils.resizeImage(height, width, 3, image,
      DeepImageFeaturizer.scaleHints(getScaleHint)), imSchema)

    val imageDF = dataFrame
      .withColumn(RESIZED_IMAGE_COL, resizeUdf(col(getInputCol)))
      .withColumn(INPUT_BUFFER_COL, col(s"$RESIZED_IMAGE_COL.data"))

    val shapeHints = new ShapeDescription(
      out = Map.empty[String, Shape],
      requestedFetches = Seq(model.graphOutputNode),
      inputs = Map(model.graphInputNode -> INPUT_BUFFER_COL)
    )

    val toVector = udf { features: Seq[Double] =>
      Vectors.dense(features.toArray)
    }

    DebugRowOps
      .mapRows(imageDF, model.graph, shapeHints)
      .withColumn(getOutputCol, toVector(col(model.graphOutputNode)))
      .drop(model.graphOutputNode, RESIZED_IMAGE_COL, INPUT_BUFFER_COL)
  }
}

object DeepImageFeaturizer extends DefaultParamsReadable[DeepImageFeaturizer] {
  /**
   * The deep image featurizer uses the information provided by named Image model to apply the
   * tensorflow graph, given in NamedImageModel.graph as a GraphDef, to an image column of a
   * dataframe, represented using ImageSchema.columnSchema.
   *
   * For DeepImageFeaturizer to apply a graph, the graph must 1) have exactly 1 input Tensor of
   * shape () and type String. The binary data field of the image will be passed to this input
   * node. Also the graph should 2) have exactly 1 output Tensor. This output tensor should be 1d
   * and be of type Double. Images will be resized before being passed to the graph so the graph
   * is guaranteed to receive an binary input of length (height * width * 3). The channel order
   * is generally BGR, but it is up to the user to ensure that the input data channel order
   * matches the channel order of the graph being applied.
   *
   * Note:
   * The graphOutputNode will be used as a temporary column on the dataframe so whenever possible,
   * use a unique name to avoid column name conflicts.
   *
   */

  // TODO: support batched graphs with mapBlocks

  private[sparkdl] trait NamedImageModel {
    def name: String
    def height: Int
    def width: Int
    def graph: GraphDef
    def graphInputNode: String
    def graphOutputNode: String
  }

  /**
   * A map to help us get the model object based on it's name.
   */
  private val supportedModelMap: Map[String, NamedImageModel] = {
    val empty = Map.empty[String, NamedImageModel]
    Models._supportedModels.foldLeft(empty){ case (map, model) => map.updated(model.name, model) }
  }

  val scaleHints: Map[String, Int] = Map(
    "SCALE_AREA_AVERAGING" -> java.awt.Image.SCALE_AREA_AVERAGING,
    "SCALE_DEFAULT" -> java.awt.Image.SCALE_DEFAULT,
    "SCALE_FAST" -> java.awt.Image.SCALE_FAST,
    "SCALE_REPLICATE" -> java.awt.Image.SCALE_REPLICATE,
    "SCALE_SMOOTH" -> java.awt.Image.SCALE_SMOOTH
  )
  def scaleHintsJava = scaleHints.asJava
  /**
   * The valid values that can be used to set the "modelName" param.
   */
  val supportedModels: Set[String] = supportedModelMap.keySet.filter(! _.startsWith("_"))
}
