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

import org.tensorflow.{Graph => TFGraph, Session => TFSession}
import org.tensorframes.{dsl => tf}

trait GraphScoping { self: FunSuite =>
  import tf.withGraph

  def testGraph(banner: String)(block: => Unit): Unit = {
    test(s"[tfrm:sql-udf-impl] $banner") { withGraph { block } }
  }

  // Provides both a TensoFlow Graph and Session
  def testIsolatedSession(banner: String)(block: (TFGraph, TFSession) => Unit): Unit = {
    test(s"[tf:iso-sess] $banner") {
      val g = new TFGraph()
      val sess = new TFSession(g)
      block(g, sess)
    }
  }

  // Following TensorFlow's Java API example
  // Reference: tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java
  def testGraphBuilder(banner: String)(block: GraphBuilder => Unit): Unit = {
    test(s"[tf:iso-sess] $banner") {
      val builder = new GraphBuilder(new TFGraph())
      block(builder)
      builder.close()
    }
  }

}
