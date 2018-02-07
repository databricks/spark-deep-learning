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

import java.awt.{Color, Image}
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

import scala.util.Random

import org.scalatest.FunSuite

import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.sql.Row

object ImageUtilsSuite {

  /** Read image data into a BufferedImage, then use our utility method to convert to a row image */
  def getImageRow(resourcePath: String): Row = {
    val resourceStream = getClass.getResourceAsStream(resourcePath)
    val imageBuffer = ImageIO.read(resourceStream)
    ImageUtils.spImageFromBufferedImage(imageBuffer)
  }

  def smallerImage: Row = getImageRow("/sparkdl/00081101-small-version.png")
  def biggerImage: Row = getImageRow("/sparkdl/test-image-collection/00081101.jpg")
}

class ImageUtilsSuite extends FunSuite {
  // We want to make sure to test ImageUtils in headless mode to ensure it'll work on all systems.
  assert(System.getProperty("java.awt.headless") === "true")

  import ImageUtilsSuite._

  test("Test spImage resize.") {
    def javaResize(imagePath: String, tgtWidth: Int, tgtHeight: Int): Row = {
      // Read BufferedImage directly from file
      val resourceStream = getClass.getResourceAsStream(imagePath)
      val srcImg = ImageIO.read(resourceStream)
      val tgtImg = new BufferedImage(tgtWidth, tgtHeight, srcImg.getType)
      // scaledImg is a java.awt.Image which supports drawing but not pixel lookup by index.
      val scaledImg = srcImg.getScaledInstance(tgtWidth, tgtHeight, Image.SCALE_AREA_AVERAGING)
      // Draw scaledImage onto resized (usually smaller) tgtImg so extract individual pixel values.
      val graphic = tgtImg.createGraphics()
      graphic.drawImage(scaledImg, 0, 0, null)
      graphic.dispose()
      ImageUtils.spImageFromBufferedImage(tgtImg)
    }

    for (channels <- Seq(1, 3, 4)) {
      val path = s"/sparkdl/test-image-collection/${channels}_channels/00074201.png"
      val biggerImage = getImageRow(path)
      val tgtHeight: Int = ImageSchema.getHeight(biggerImage) / 2
      val tgtWidth: Int = ImageSchema.getWidth(biggerImage) / 2
      val tgtChannels: Int = ImageSchema.getNChannels(biggerImage)

      val expectedImage = javaResize(path, tgtWidth, tgtHeight)
      val resizedImage = ImageUtils.resizeImage(tgtHeight, tgtWidth, tgtChannels, biggerImage)
      assert(resizedImage === expectedImage, "Resizing image did not produce expected smaller " +
        "image.")
    }
  }

  test ("Test Row image -> BufferedImage -> Row image") {
    val height = 200
    val width = 100
    for (channels <- Seq(1, 3, 4)) {
      val rand = new Random(971)
      val imageData = Array.ofDim[Byte](height * width * channels)
      rand.nextBytes(imageData)
      val ocvType = s"CV_8UC$channels"
      val spImage = Row(null, height, width, channels, ImageSchema.ocvTypes(ocvType), imageData)
      val bufferedImage = ImageUtils.spImageToBufferedImage(spImage)
      val testImage = ImageUtils.spImageFromBufferedImage(bufferedImage)
      assert(spImage === testImage, "Image changed during conversion")
    }
  }

  test("Simple BufferedImage from Row Image") {
    val height = 2
    val width = 5
    for (channels <- Seq(3, 4)) {
      val rawData: Array[Byte] = (0 until height).flatMap { i =>
        (0 until width).flatMap { j =>
          // Generate data for the pixel at (i, j).
          // We set the value of channel c to (i * 100 + j * 10 + c)
          Range(0, channels).map { c =>
            i * 100 + j * 10 + c
          }.toSeq
        }
      }.map(_.toByte).toArray

      val spImage = Row(null, height, width, channels, ImageSchema.ocvTypes(s"CV_8UC$channels"),
        rawData)
      val bufferedImage = ImageUtils.spImageToBufferedImage(spImage)
      val hasAlpha = bufferedImage.getColorModel.hasAlpha
      for (h <- 0 until height) {
        for (w <- 0 until width) {
          val rgb = bufferedImage.getRGB(w, h)
          val color = new Color(rgb, hasAlpha)
          assert(color.getBlue.toByte === (h * 100 + w * 10 ).toByte)
          assert(color.getGreen.toByte === (h * 100 + w * 10 + 1).toByte)
          assert(color.getRed.toByte === (h * 100 + w * 10 + 2).toByte)
          if (channels == 4) {
            assert(color.getAlpha.toByte === (h * 100 + w * 10 + 3).toByte)
          }
        }
      }
    }
  }

  test("Simple BufferedImage from Row Image: grayscale") {
    val height = 20
    val width = 10
    val rawData: Array[Byte] = Range(0, height * width).map(_.toByte).toArray
    val spImage = Row(null, height, width, 1, ImageSchema.ocvTypes("CV_8UC1"), rawData)
    val bufferedImage = ImageUtils.spImageToBufferedImage(spImage)
    val raster = bufferedImage.getRaster
    for (h <- 0 until height) {
      for (w <- 0 until width) {
        assert(raster.getSample(w, h, 0) == h * width + w)
      }
    }
  }
}
