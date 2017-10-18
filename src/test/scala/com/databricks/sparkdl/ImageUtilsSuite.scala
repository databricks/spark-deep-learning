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

import java.nio.file.{Files, Paths}

import org.apache.spark.sql.Row
import org.scalatest.FunSuite

object ImageUtilsSuite {
  val SRC_HEIGHT = 500
  val SRC_WIDTH = 336
  val TGT_HEIGHT = 60
  val TGT_WIDTH = 40
  val SRC_CHANNELS, TGT_CHANNELS = 3

  def sourceTestImages() = {
    val cls = getClass()
    val biggerImage = Files.readAllBytes(Paths.get(cls.getResource("/biggerImage.raw").getFile))
    assert(
      biggerImage.length == SRC_CHANNELS * SRC_HEIGHT * SRC_WIDTH,
      "Something seems wrong with the image buffer.")
    val smallerImage = Files.readAllBytes(Paths.get(cls.getResource("/smallerImage.raw").getFile))
    assert(
      smallerImage.length == TGT_CHANNELS * TGT_HEIGHT * TGT_WIDTH,
      "Something seems wrong with the image buffer.")
    (biggerImage, smallerImage)
  }
}

class ImageUtilsSuite extends FunSuite {
  import ImageUtilsSuite._

  test("test binary image resize") {

    val (biggerImage, smallerImage) = sourceTestImages()
    val resizedImage = ImageUtils.resizeImage(
      SRC_HEIGHT,
      SRC_WIDTH,
      SRC_CHANNELS,
      TGT_HEIGHT,
      TGT_WIDTH,
      TGT_CHANNELS,
      biggerImage)
    assert(resizedImage.deep == smallerImage.deep, "resizeImage did not give expected result.")
  }

  test("test image resize udf helper") {
    val (biggerImage, smallerImage) = sourceTestImages()
    val imageResizer = ImageUtils.resizeSPImage(TGT_HEIGHT, TGT_WIDTH, TGT_CHANNELS)(_)
    val imageAsRow = Row(null, SRC_HEIGHT, SRC_WIDTH, SRC_CHANNELS, "CV_U83C", biggerImage)
    val resizedImage = imageResizer(imageAsRow)
    val resizedBuffer = resizedImage.getAs[Array[Byte]](5)
    assert(resizedBuffer.deep == smallerImage.deep, "resizeImage did not give expected result.")

    val reResizedImage = imageResizer(resizedImage)
    // We do an explicit reference equality here because src & tgt sizes match and short circuit.
    assert(reResizedImage == resizedImage, "resizeImage did not give expected result.")
  }

}
