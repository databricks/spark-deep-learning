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

import java.awt.image.BufferedImage
import java.awt.{Color, Image}

import org.apache.spark.sql.Row

object ImageUtils {

  /**
   * Resizes and image and returns it as an Array[Byte]. Only 1 and 3 channel inputs, where each
   * channel is a single Byte, are currently supported.
   *
   * @param srcHeight height of input image
   * @param srcWidth width of input image
   * @param srcChannels number of channels for input image (1 or 3)
   * @param tgtHeight desired height of output image
   * @param tgtWidth desired width of output image
   * @param tgtChannels number of channels of output image (must be 3), may be used later to
   *                    support more channels.
   * @param imageData buffer containing image data.
   * @return Array[Byte] of image data, if the input was BGR or 1 channel, the output will be BGR.
   */
  private[sparkdl] def resizeImage(
      srcHeight: Int,
      srcWidth: Int,
      srcChannels: Int,
      tgtHeight: Int,
      tgtWidth: Int,
      tgtChannels: Int,
      imageData: Array[Byte]) = {
    require(tgtChannels == 3, "output must have 3 channels.")
    val srcImg = new BufferedImage(srcHeight, srcWidth, BufferedImage.TYPE_3BYTE_BGR)

    var offset, h = 0
    while (h < srcHeight) {
      var w = 0
      while (w < srcWidth) {
        val (r, g, b): (Byte, Byte, Byte) = srcChannels match {
          case 1 =>
            val i = imageData(offset)
            (i, i, i)
          case 3 =>
            val b = imageData(offset)
            val g = imageData(offset + 1)
            val r = imageData(offset + 2)
            (r, g, b)
          case _ =>
            require(false, "`srcChannels` must be 1 or 3.")
            // make type check happy
            (0, 0, 0)
        }

        val color = new Color(r & 0xff, g & 0xff, b & 0xff)
        srcImg.setRGB(h, w, color.getRGB)
        offset += srcChannels
        w += 1
      }
      h += 1
    }

    val scaledImg = srcImg.getScaledInstance(tgtHeight, tgtWidth, Image.SCALE_AREA_AVERAGING)
    val tgtImg = new BufferedImage(tgtHeight, tgtWidth, BufferedImage.TYPE_3BYTE_BGR)

    // Draw the image on to the buffered image
    val graphic = tgtImg.createGraphics()
    graphic.drawImage(scaledImg, 0, 0, null)
    graphic.dispose()

    val decoded = new Array[Byte](tgtHeight * tgtWidth * tgtChannels)
    offset = 0
    h = 0
    while (h < tgtHeight) {
      var w = 0
      while (w < tgtWidth) {
        val color = new Color(tgtImg.getRGB(h, w))
        decoded(offset) = color.getBlue.toByte
        decoded(offset + 1) = color.getGreen.toByte
        decoded(offset + 2) = color.getRed.toByte

        offset += tgtChannels
        w += 1
      }
      h += 1
    }
    decoded
  }

  /**
   * Resizes images given in Rows representation with ImageSchema.columnSchema
   *
   * Can be used to create a udf to resize images, for example:
   *   val myResizeUdf = udf(resizeSPImage(height, width, 3) _, ImageSchema.columnSchema)
   *
   * @param tgtHeight Desired height of the image.
   * @param tgtWidth Desired width of the image.
   * @param tgtChannels Target number of channels (must be 3 for now).
   * @param r Image to resize.
   * @return Row, The resized image.
   */
  def resizeSPImage(tgtHeight: Int, tgtWidth: Int, tgtChannels: Int)(r: Row): Row = {
    val height = r.getInt(1)
    val width = r.getInt(2)
    val nChannels = r.getInt(3)
    if ((nChannels == tgtChannels) && (height == tgtHeight) && (width == tgtWidth)) {
      r
    } else {
      val data = r.getAs[Array[Byte]](5)
      val resized = resizeImage(height, width, nChannels, tgtHeight, tgtWidth, tgtChannels, data)
      // TODO: udpate mode to be Int when spark.image is merged.
      val mode = "CV_8UC3"
      Row(null, tgtHeight, tgtWidth, tgtChannels, mode, resized)
    }
  }
}
