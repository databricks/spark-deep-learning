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

import java.awt.color.ColorSpace
import java.awt.image.BufferedImage
import java.awt.{Color, Image}

import org.apache.spark.ml.image.ImageSchema
import org.apache.spark.sql.Row

private[sparkdl] object ImageUtils {

  /**
   * Takes a Row image (spImage) and returns a Java BufferedImage. Currently supports 1, 3, & 4
   * channel images. If the Row image has 3 or 4 channels, we assume the channels are in BGR(A)
   * order.
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

    val image = channels match {
      case 1 => new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
      case 3 => new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
      case 4 => new BufferedImage(width, height, BufferedImage.TYPE_4BYTE_ABGR)
    }

    val isGray = image.getColorModel.getColorSpace.getType == ColorSpace.TYPE_GRAY
    val hasAlpha = image.getColorModel.hasAlpha
    var offset = 0
    // Grayscale images in Java require special handling to get the correct intensity
    if (isGray) {
      val raster = image.getRaster
      for (h <- 0 until height) {
        for (w <- 0 until width) {
          raster.setSample(w, h, 0, imageData(offset))
          offset += 1
        }
      }
    } else {
      for (h <- 0 until height) {
        for (w <- 0 until width) {
          val b = imageData(offset)
          val g = imageData(offset + 1)
          val r = imageData(offset + 2)
          val color = if (hasAlpha) {
            val a = imageData(offset + 3)
            new Color(r & 0xff, g & 0xff, b & 0xff, a & 0xff)
          } else {
            new Color(r & 0xff, g & 0xff, b & 0xff)
          }
          image.setRGB(w, h, color.getRGB)
          offset += channels
        }
      }
    }

    image
  }

  /** Returns the number of channels in the passed-in buffered image. */
  private def getNumChannels(img: BufferedImage): Int = {
    val isGray = img.getColorModel.getColorSpace.getType == ColorSpace.TYPE_GRAY
    val hasAlpha = img.getColorModel.hasAlpha
    if (isGray) {
      1
    } else if (hasAlpha) {
      4
    } else {
      3
    }
  }

  /** Returns the OCV type (int) of the passed-in image */
  private def getOCVType(img: BufferedImage): Int = {
    val isGray = img.getColorModel.getColorSpace.getType == ColorSpace.TYPE_GRAY
    val hasAlpha = img.getColorModel.hasAlpha
    if (isGray) {
      ImageSchema.ocvTypes("CV_8UC1")
    } else if (hasAlpha) {
      ImageSchema.ocvTypes("CV_8UC4")
    } else {
      ImageSchema.ocvTypes("CV_8UC3")
    }
  }

  /**
   * Takes a Java BufferedImage and returns a Row Image (spImage).
   *
   * @param image Java BufferedImage.
   * @return Row image in spark.ml.image format with channels in BGR(A) order.
   */
  private[sparkdl] def spImageFromBufferedImage(image: BufferedImage, origin: String = null): Row = {
    val channels = getNumChannels(image)
    val height = image.getHeight
    val width = image.getWidth
    val nChannels = getNumChannels(image)
    val isGray = image.getColorModel.getColorSpace.getType == ColorSpace.TYPE_GRAY
    val hasAlpha = image.getColorModel.hasAlpha

    val decoded = new Array[Byte](height * width * channels)
    var offset, h = 0

    // Grayscale images in Java require special handling to get the correct intensity
    if (isGray) {
      var offset = 0
      val raster = image.getRaster
      for (h <- 0 until height) {
        for (w <- 0 until width) {
          decoded(offset) = raster.getSample(w, h, 0).toByte
          offset += 1
        }
      }
    } else {
      var offset = 0
      for (h <- 0 until height) {
        for (w <- 0 until width) {
          val color = new Color(image.getRGB(w, h), hasAlpha)
          decoded(offset) = color.getBlue.toByte
          decoded(offset + 1) = color.getGreen.toByte
          decoded(offset + 2) = color.getRed.toByte
          if (hasAlpha) {
            decoded(offset + 3) = color.getAlpha.toByte
          }
          offset += nChannels
        }
      }
    }

    Row(origin, height, width, channels, getOCVType(image), decoded)

  }

  /**
   * Resizes an image and returns it as an Array[Byte]. Only 1, 3, and 4 channel inputs, where each
   * channel is a single Byte, are currently supported. Only BGR channel order is supported but
   * this might work for other channel orders.
   *
   * @param tgtHeight   desired height of output image.
   * @param tgtWidth    desired width of output image.
   * @param tgtChannels number of channels in output image.
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
    val height = ImageSchema.getHeight(spImage)
    val width = ImageSchema.getWidth(spImage)
    val nChannels = ImageSchema.getNChannels(spImage)

    if ((nChannels == tgtChannels) && (height == tgtHeight) && (width == tgtWidth)) {
      spImage
    } else {
      val srcImg = spImageToBufferedImage(spImage)
      val tgtImgType = tgtChannels match {
        case 1 => BufferedImage.TYPE_BYTE_GRAY
        case 3 => BufferedImage.TYPE_3BYTE_BGR
        case 4 => BufferedImage.TYPE_4BYTE_ABGR
      }
      val tgtImg = new BufferedImage(tgtWidth, tgtHeight, tgtImgType)
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
