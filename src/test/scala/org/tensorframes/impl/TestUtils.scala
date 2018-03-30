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

import java.nio.file.{Files, Paths => JPaths}

import scala.collection.JavaConverters._

import org.tensorflow.framework.GraphDef
import org.tensorflow.{Tensor, Graph => TFGraph, Output => TFOut, Session => TFSession}
import org.tensorframes.ShapeDescription



/**
 * Utilities for buidling graphs with TensorFlow Java API
 * 
 * Reference: tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
 */
class GraphBuilder(g: TFGraph) {

  var varIdx: Long = 0L  
  @transient private[this] var _sess: Option[TFSession] = None
  lazy val sess: TFSession = {
    if (_sess.isEmpty) {
      _sess = Some(new TFSession(g))
    }
    _sess.get
  }

  def close(): Unit = {
    _sess.foreach(_.close())
    g.close()
  }

  def op(opType: String, name: Option[String] = None)(in0: TFOut[_], ins: TFOut[_]*): TFOut[_] = {
    val opName = name.getOrElse(s"$opType-${varIdx += 1}")
    var b = g.opBuilder(opType, opName).addInput(in0)
    ins.foreach { in => b = b.addInput(in) }
    b.build().output(0)
  }  

  def const[T](name: String, value: T): TFOut[_] = {
    val tnsr = Tensor.create(value)
    g.opBuilder("Const", name)
      .setAttr("dtype", tnsr.dataType())
      .setAttr("value", tnsr)
      .build().output(0)
  }

  def run(feeds: Map[String, Any], fetch: String): Tensor[_] = {
    run(feeds, Seq(fetch)).head
  }

  def run(feeds: Map[String, Any], fetches: Seq[String]): Seq[Tensor[_]] = {
    var runner = sess.runner()
    feeds.foreach { 
      case (name, tnsr: Tensor[_]) =>
        runner = runner.feed(name, tnsr)
      case (name, value) =>
        runner = runner.feed(name, Tensor.create(value))
    }
    fetches.foreach { name => runner = runner.fetch(name) }
    runner.run().asScala
  }
}

/**
 * Utilities for building graphs with TensorFrames API (with DSL)
 * 
 * TODO: these are taken from TensorFrames, we will eventually merge them
 */
private[tensorframes] object TestUtils {

  import org.tensorframes.dsl._

  def buildGraph(node: Operation, nodes: Operation*): GraphDef = {
    buildGraph(Seq(node) ++ nodes)
  }

  def loadGraph(file: String): GraphDef = {
    val byteArray = Files.readAllBytes(JPaths.get(file))
    GraphDef.newBuilder().mergeFrom(byteArray).build()
  }

  def analyzeGraph(nodes: Operation*): (GraphDef, Seq[GraphNodeSummary]) = {
    val g = buildGraph(nodes.head, nodes.tail: _*)
    g -> TensorFlowOps.analyzeGraphTF(g, extraInfo(nodes))
  }

  // Implicit type conversion
  implicit def op2Node(op: Operation): Node = op.asInstanceOf[Node]
  implicit def ops2Nodes(ops: Seq[Operation]): Seq[Node] = ops.map(op2Node)

  private def getClosure(node: Node, treated: Map[String, Node]): Map[String, Node] = {
    val explored = node.parents
      .filterNot(n => treated.contains(n.name))
      .flatMap(getClosure(_, treated + (node.name -> node)))
      .toMap

    uniqueByName(node +: (explored.values.toSeq ++ treated.values.toSeq))
  }

  private def uniqueByName(nodes: Seq[Node]): Map[String, Node] = {
    nodes.groupBy(_.name).mapValues(_.head)
  }

  def buildGraph(nodes: Seq[Operation]): GraphDef = {
    nodes.foreach(_.freeze())
    nodes.foreach(_.freeze(everything=true))
    var treated: Map[String, Node] = Map.empty
    nodes.foreach { node =>
      treated = getClosure(node, treated)
    }
    val b = GraphDef.newBuilder()
    treated.values.flatMap(_.nodes).foreach(b.addNode)
    b.build()
  }

  private def extraInfo(fetches: Seq[Node]): ShapeDescription = {
    val m2 = fetches.map(n => n.name -> n.name).toMap
    ShapeDescription(
      fetches.map(n => n.name -> n.shape).toMap,
      fetches.map(_.name),
      m2)
  }
}
