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

import java.util.concurrent.atomic.AtomicInteger

import scala.collection.JavaConverters._

import org.tensorflow.{Graph, Session}
import org.tensorflow.framework.GraphDef

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{Column, Row}
import org.apache.spark.sql.functions.struct
import org.apache.spark.sql.sparkdl_stubs.{PipelinedUDF, RowUDF}
import org.apache.spark.sql.types.{StructType, DataType}
import org.apache.spark.sql.catalyst.expressions.Expression

import org.tensorframes.{ColumnInformation, Shape, ShapeDescription, _}


/**
  * Making and running TensorFlow based UserDefinedFunctions (UDF).
  * Column transforms. These are not as efficient as working with full dataframes,
  * but they may be simpler to work with.
  */
// TODO: this is mostly cut and paste from performMapRows -> should combine
object SqlOps extends Logging {
  import SchemaTransforms.{get, check}

  // Counter: the number of sessions currently opened.
  private class LocalState(
      val session: Session,
      val graphHash: Int,
      val graph: Graph,
      val counter: AtomicInteger) {
    def close(): Unit = {
      session.close()
      graph.close()
    }
  }

  // A map of graph hash -> state for this graph.
  private[this] var current: Map[Int, LocalState] = Map.empty
  private[this] val lock = new Object()

  // The maximum number of sessions that can be opened concurrently.
  // TODO: investigate TensorFlow's parallel execution
  // TensorFlow's session run can be executed concurrently: 
  //   The Session API allows multiple concurrent steps (i.e. calls to tf.Session.run) in parallel. 
  //   This enables the runtime to get higher throughput, 
  //   if a single step does not use all of the resources in your computer.
  // (from https://www.tensorflow.org/programmers_guide/faq)
  // TODO: make this parameter configurable
  val maxSessions: Int = 10

  /**
   * Experimental: expresses a TensorFlow Row transform as a SQL-registrable UDF.
   *
   * This is not as efficient as doing a direct dataframe transform, and it leaks
   * some resources after the completion of the transform. These resources may easily be
   * be reclaimed though.
   *
   * @param udfName: the name of the UDF that is going to be presented when building the udf.
   * @param graph the graph of the computation
   * @param shapeHints the extra info regarding the shapes
   * @param applyBlocks: if true, the graph is assumed to accepted vectorized inputs.
   * @param flattenStruct: if true, and if the return type contains a single field, then
   *                     this field will be exposed as the value, instead of returning a struct.
   */
  def makeUDF(
      udfName: String,
      graph: GraphDef,
      shapeHints: ShapeDescription,
      applyBlocks: Boolean,
      flattenStruct: Boolean): UserDefinedFunction = {

    val (outputSchema, udf1) = makeUDF0(udfName, graph, shapeHints, applyBlocks = applyBlocks)
    outputSchema match {
      case StructType(Array(f1)) if flattenStruct =>
        // Flatten the structure around the field
        // For now, the only reasonable way is to create another UDF :(
        def fun(r: Row): Any = r.get(0)
        val udf2 = UserDefinedFunction(fun _, f1.dataType, None)
        PipelinedUDF(udfName, udf1, udf2)
      case _ =>
        udf1
    }
  }

  /**
   * Experimental: expresses a Row transform as a SQL-registrable UDF.
   *
   * This is not as efficient as doing a direct dataframe transform, and it leaks
   * some resources after the completion of the transform. These resources may easily be
   * be reclaimed though.
   *
   * @param udfName: the name of the UDF that is going to be presented when building the udf.
   * @param graph the graph of the computation
   * @param shapeHints the extra info regarding the shapes
   * @param applyBlocks: if true, the graph is assumed to accepted vectorized inputs.
   *
   * Returns the UDF and the schema of the output (always a struct)
   */
  def makeUDF0(
      udfName: String,
      graph: GraphDef,
      shapeHints: ShapeDescription,
      applyBlocks: Boolean): (StructType, UserDefinedFunction) = {
    val summary = TensorFlowOps.analyzeGraphTF(graph, shapeHints)
      .map(x => x.name -> x).toMap
    val inputs = summary.filter(_._2.isInput)
    val outputs = summary.filter(_._2.isOutput)

    // The output schema of the block from the data generated by TF.
    val outputTFSchema: StructType = {
      // The order of the output columns is decided for now by their names.
      val fields = outputs.values.toSeq.sortBy(_.name).map { out =>
        // Compute the shape of the block. If the data is blocked, there is no need to append an extra dimension.
        val blockShape = if (applyBlocks) { out.shape } else {
          // The shapes we get in each output node are the shape of the cells of each column, not the
          // shape of the column. Add Unknown since we do not know the exact length of the block.
          out.shape.prepend(Shape.Unknown)
        }
        ColumnInformation.structField(out.name, out.scalarType, blockShape)
      }
      StructType(fields.toArray)
    }

    val outputSchema: StructType = StructType(outputTFSchema)

    def processColumn(inputColumn: Column): Any => Row = {
      // Special case: if the column has a single non-structural field and if
      // there is a single input in the graph, we automatically wrap the input in a structure.
      (inputs.keySet.toSeq, inputColumn.expr.dataType) match {
        case (_, _: StructType) => processColumn0(inputColumn)
        case (Seq(name1), _) => processColumn0(struct(inputColumn.alias(name1)))
        case (names, dt) =>
          throw new Exception(s"Too many graph inputs for the given column type: names=$names, dt=$dt")
      }
    }

    def processColumn0(inputColumn: Column): Any => Row = {
      val inputSchema = inputColumn.expr.dataType match {
        case st: StructType => st
        case x: Any => throw new Exception(
          s"Only structures are currently accepted: given $x")
      }
      val fieldsByName = inputSchema.fields.map(f => f.name -> f).toMap
      val cols = inputSchema.fieldNames.mkString(", ")

      // Initial check of the input.
      inputs.values.foreach { in =>
        val fname = get(shapeHints.inputs.get(in.name),
          s"The graph placeholder ${in.name} was not given a corresponding dataframe field name as input:" +
            s"hints: ${shapeHints.inputs}")

        val f = get(fieldsByName.get(fname),
          s"Graph input ${in.name} found, but no column to match it. Dataframe columns: $cols")

        val stf = get(ColumnInformation(f).stf,
          s"Data column ${f.name} has not been analyzed yet, cannot run TF on this dataframe")

        check(stf.dataType == in.scalarType,
          s"The type of node '${in.name}' (${stf.dataType}) is not compatible with the data type " +
            s"of the column (${in.scalarType})")

        val cellShape = stf.shape.tail
        if (applyBlocks) {
          check(in.shape.numDims >= 1,
          s"The input '${in.name}' is expected to at least a vector, but it currently scalar")
          // Check against the tail (which should be the cell).
          check(cellShape.checkMorePreciseThan(in.shape.tail),
            s"The data column '${f.name}' has shape ${stf.shape} (not compatible) with shape" +
              s" ${in.shape} requested by the TF graph")
        } else {
          val cellShape = stf.shape.tail
          // TODO: UNCOMMENT this
          // // No check for unknowns: we allow unknowns in the first dimension of the cell shape.
          // check(cellShape.checkMorePreciseThan(in.shape),
          //   s"The data column '${f.name}' has shape ${stf.shape} (not compatible) with shape" +
          //     s" ${in.shape} requested by the TF graph")
        }

        check(in.isPlaceholder,
          s"Invalid type for input node ${in.name}. It has to be a placeholder")
      }

      // The column indices requested by TF, and the name of the placeholder that gets fed.
      val requestedTFInput: Array[(NodePath, Int)] = {
        val colIdxs = inputSchema.fieldNames.zipWithIndex.toMap
        inputs.keys.map { nodePath =>
          val fieldName = shapeHints.inputs(nodePath)
          nodePath -> colIdxs(fieldName)
        }   .toArray
      }

      logger.debug(s"makeUDF: input schema = $inputSchema, requested cols: ${requestedTFInput.toSeq}" +
        s" complete output schema = $outputSchema")
      // TODO: this is leaking the file.
      val sc = SparkContext.getOrCreate()
      val gProto = sc.broadcast(TensorFlowOps.graphSerial(graph))
      val f = performUDF(inputSchema, requestedTFInput, gProto, outputTFSchema, applyBlocks)
      f
    }

    outputSchema -> new RowUDF(udfName, processColumn _, outputSchema)
  }

  def performUDF(
    inputSchema: StructType,
    inputTFCols: Array[(NodePath, Int)],
    g_bc: Broadcast[SerializedGraph],
    tfOutputSchema: StructType,
    applyBlocks: Boolean): Any => Row = {

    logger.debug(s"performUDF: inputSchema=$inputSchema inputTFCols=${inputTFCols.toSeq}")

    def f(in: Any): Row = {
      val row = in match {
        case r: Row => r
        case x => Row(x)
      }
      val g = g_bc.value
      retrieveSession(g) { session =>
        g.evictContent()
        val inputTensors = TFDataOps.convert(row, inputSchema, inputTFCols)
        logger.debug(s"performUDF:inputTensors=$inputTensors")
        val requested = tfOutputSchema.map(_.name)
        var runner = session.runner()
        for (req <- requested) {
          runner = runner.fetch(req)
        }
        for ((inputName, inputTensor) <- inputTensors) {
          runner = runner.feed(inputName, inputTensor)
        }
        val outs = runner.run().asScala
        logger.debug(s"performUDF:outs=$outs")
        // Close the inputs
        inputTensors.map(_._2).foreach(_.close())
        val res = TFDataOps.convertBack(outs, tfOutputSchema, Array(row), inputSchema, appendInput = false)
        // Close the outputs
        outs.foreach(_.close())
        assert(res.hasNext)
        val r = res.next()
        assert(!res.hasNext)
        //      logger.debug(s"performUDF: r=$r")
        r
      }
    }
    f
  }

  private def retrieveSession[T](g: SerializedGraph)(f: Session => T): T = {
    //  // This is a better version that uses the hash of the content, but it requires changes to
    //  // TensorFrames. Only use with version 0.2.9+
    val hash = java.util.Arrays.hashCode(g.content)
    retrieveSession(g, hash, f)
  }

  private def retrieveSession[T](g: SerializedGraph, gHash: Int, f: Session => T): T = {
    // Do some cleanup first:
    lock.synchronized {
      val numberOfSessionsToClose = Math.max(current.size - maxSessions, 0)
      // This is best effort only, there may be more sessions opened at some point.
      if (numberOfSessionsToClose > 0) {
        // Find some sessions to close: they are not currently used, and they are not the requested session.
        val sessionsToRemove = current.valuesIterator
          .filter { s => s.counter.get() == 0 && s.graphHash != gHash }
          .take(numberOfSessionsToClose)
        for (state <- sessionsToRemove) {
          logger.debug(s"Removing session ${state.graphHash}")
          state.close()
          current = current - state.graphHash
        }
      }
    }

    // Now, try to retrieve the session, or create a new one.
    // TODO: use a double lock mechanism or a lazy value, since importing a graph may take a long time.
    val state = lock.synchronized {
      val state0 = current.get(gHash) match {
        case None =>
          // Add a new session
          val tg = new Graph()
          tg.importGraphDef(g.content)
          val s = new Session(tg)
          val ls = new LocalState(s, gHash, tg, new AtomicInteger(0))
          current = current + (gHash -> ls)
          ls
        case Some(ls) =>
          // Serve the existing session
          ls
      }
      // Increment the counter in the locked section, to guarantee that the session does not get collected.
      state0.counter.incrementAndGet()
      state0
    }

    // Perform the action
    try {
      f(state.session)
    } finally {
      state.counter.decrementAndGet()
    }
  }
}
