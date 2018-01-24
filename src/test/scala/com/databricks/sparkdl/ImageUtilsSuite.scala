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

import java.awt.Color
import java.io.File
import javax.imageio.ImageIO

import scala.util.Random

import org.scalatest.FunSuite

import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.sql.Row

object ImageUtilsSuite {

  /** Read image data into a BufferedImage, then use our utility method to convert to a row image */
  def getImageRow(resourcePath: String): Row = {
    val resourceUrl = getClass.getResource(resourcePath).getFile
    val imageBuffer = ImageIO.read(new File(resourceUrl))
    ImageUtils.spImageFromBufferedImage(imageBuffer)
  }

  def smallerImage: Row = getImageRow("/sparkdl/00081101-small-version.png")
  def biggerImage: Row = getImageRow("/sparkdl/test-image-collection/00081101.jpg")
}

class ImageUtilsSuite extends FunSuite {
  // We want to make sure to test ImageUtils in headless mode to ensure it'll work on all systems.
  assert(System.getProperty("java.awt.headless") === "true")
  test("Test spImage resize.") {
    def getImagePath(imageSize: String, numChannels: Int): String = {
      s"/sparkdl/test-image-collection/${numChannels}_channels/$imageSize.png"
    }
    for (channels <- Seq(1, 3, 4)) {
      val smallerImage = ImageUtilsSuite.getImageRow(getImagePath("small", channels))
      val biggerImage = ImageUtilsSuite.getImageRow(getImagePath("big", channels))

      val tgtHeight: Int = ImageSchema.getHeight(smallerImage)
      val tgtWidth: Int = ImageSchema.getWidth(smallerImage)
      val tgtChannels: Int = ImageSchema.getNChannels(smallerImage)

      val testImage = ImageUtils.resizeImage(tgtHeight, tgtWidth, tgtChannels, biggerImage)
      assert(testImage === smallerImage, "Resizing image did not produce expected smaller image.")
    }
  }

  test ("Test Row image -> BufferedImage -> Row image") {
    val height = 200
    val width = 100
    for (channels <- Seq(3, 4)) {
      val rand = new Random(971)
      val imageData = Array.ofDim[Byte](height * width * channels)
      rand.nextBytes(imageData)
      val ocvType = s"CV_8UC$channels"
      val spImage = Row(null, height, width, channels, ImageSchema.ocvTypes(ocvType), imageData)
      val bufferedImage = ImageUtils.spImageToBufferedImage(spImage)
      val testImage = ImageUtils.spImageFromBufferedImage(bufferedImage)
      assert(spImage === testImage, s"Image changed during conversion")
    }
  }

  test("Simple BufferedImage from Row Image") {
    val height = 20
    val width = 3
    val rawData: Array[Byte] = (0 until height).flatMap { i =>
      val x = i * 10
      // B = 10 * i + w + 1
      // G = 10 * i + w + 4
      // R = 10 * i + w + 7
      // (  B      G      R,     B      G      R,     B      G      R  )
      (0 until width).flatMap { j => Seq(x + j + 1, x + j + 4, x + j + 7) }
    }.map(_.toByte).toArray

    val spImage = Row(null, height, width, 3, ImageSchema.ocvTypes("CV_8UC3"), rawData)
    val bufferedImage = ImageUtils.spImageToBufferedImage(spImage)

    for (h <- 0 until height) {
      for (w <- 0 until width) {
        val rgb = bufferedImage.getRGB(w, h)
        val color = new Color(rgb)
        assert(color.getBlue.toByte === (h * 10 + w + 1).toByte)
        assert(color.getGreen.toByte === (h * 10 + w + 4).toByte)
        assert(color.getRed.toByte === (h * 10 + w + 7).toByte)
      }
    }
  }
}
