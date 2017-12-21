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

import java.awt
import java.awt.image.BufferedImage
import java.awt.{Color, Image}

import com.sun.javafx.iio.ImageStorage.ImageType
import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

private[sparkdl] object ImageUtils {

  /**
   * Takes a Row image (spImage) and returns a Java BufferedImage. Currently supports 1 & 3
   * channel images. If the image has 3 channels, we assume the channels are in BGR order.
   *
   * @param rowImage Image in spark.ml.image format.
   * @return Java BGR BufferedImage.
   */
  private[sparkdl] def spImageToBufferedImage(rowImage: Row): BufferedImage = {
    val height = ImageSchema.getHeight(rowImage)
    val width = ImageSchema.getWidth(rowImage)
    val channels = ImageSchema.getNChannels(rowImage)
    val imageData = ImageSchema.getData(rowImage)
    require(
      imageData.length == height * width * channels,
      s"""| Only one byte per channel is currently supported, got ${imageData.length} bytes for
          | image of size ($height, $width, $channels).
       """.stripMargin
    )
    val image = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)

    var offset, h = 0
    var r, g, b: Byte = 0
    while (h < height) {
      var w = 0
      while (w < width) {
        channels match {
          case 1 =>
            b = imageData(offset)
            g = b
            r = b
          case 3 =>
            b = imageData(offset)
            g = imageData(offset + 1)
            r = imageData(offset + 2)
          case _ =>
            require(false, s"`Channels` must be 1 or 3, got $channels.")
        }

        val color = new Color(r & 0xff, g & 0xff, b & 0xff)
        image.setRGB(w, h, color.getRGB)
        offset += channels
        w += 1
      }
      h += 1
    }
    image
  }


  /**
   * Takes a Java BufferedImage and returns a Row Image (spImage).
   *
   * @param image Java BufferedImage.
   * @return Row image in spark.ml.image format with 3 channels in BGR order.
   */
  private[sparkdl] def spImageFromBufferedImage(image: BufferedImage, origin: String = null): Row = {
    val channels = 3
    val height = image.getHeight
    val width = image.getWidth

    val decoded = new Array[Byte](height * width * channels)
    var offset, h = 0
    while (h < height) {
      var w = 0
      while (w < width) {
        val color = new Color(image.getRGB(w, h))
        decoded(offset) = color.getBlue.toByte
        decoded(offset + 1) = color.getGreen.toByte
        decoded(offset + 2) = color.getRed.toByte
        offset += channels
        w += 1
      }
      h += 1
    }
    Row(origin, height, width, channels, ImageSchema.ocvTypes("CV_8UC3"), decoded)
  }

  /**
   * Resizes an image and returns it as an Array[Byte]. Only 1 and 3 channel inputs, where each
   * channel is a single Byte, are currently supported. Only BGR channel order is supported but
   * this might work for other channel orders.
   *
   * @param tgtHeight   desired height of output image.
   * @param tgtWidth    desired width of output image.
   * @param tgtChannels number of channels of output image (must be 3), may be used later to
   *                    support more channels.
   * @param spImage     image to resize.
   * @param scaleHint   hint which algorhitm to use, see java.awt.Image#SCALE_SCALE_AREA_AVERAGING
   * @return resized image, if the input was BGR or 1 channel, the output will be BGR.
   */
  private[sparkdl] def resizeImage(
    tgtHeight: Int,
    tgtWidth: Int,
    tgtChannels: Int,
    spImage: Row,
    scaleHint: Int = Image.SCALE_AREA_AVERAGING): Row = {
    require(tgtChannels == 3, s"`tgtChannels` was set to $tgtChannels, must be 3.")

    val height = ImageSchema.getHeight(spImage)
    val width = ImageSchema.getWidth(spImage)
    val nChannels = ImageSchema.getNChannels(spImage)

    if ((nChannels == tgtChannels) && (height == tgtHeight) && (width == tgtWidth)) {
      spImage
    } else {
      val srcImg = spImageToBufferedImage(spImage)
      val tgtImg = new BufferedImage(tgtWidth, tgtHeight, BufferedImage.TYPE_3BYTE_BGR)
      // scaledImg is a java.awt.Image which supports drawing but not pixel lookup by index.
      val scaledImg = srcImg.getScaledInstance(tgtWidth, tgtHeight, scaleHint)
      // Draw scaledImage onto resized (usually smaller) tgtImg so extract individual pixel values.
      val graphic = tgtImg.createGraphics()
      graphic.drawImage(scaledImg, 0, 0, null)
      graphic.dispose()
      spImageFromBufferedImage(tgtImg, origin=ImageSchema.getOrigin(spImage))
    }
  }
}
