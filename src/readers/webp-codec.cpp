/**
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This library provides functionality to convert, manipulate, and process
 * 3D Gaussian splatting data formats used in real-time neural rendering.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * splat is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * For more information, visit the project's homepage or contact the author.
 */

#include <splat/readers/webp-codec.h>
#include <webp/decode.h>
#include <webp/encode.h>

#include <stdexcept>

namespace splat {

DecodedImage decodeRGBA(const uint8_t* data, size_t size) {
  int width, height;
  uint8_t* output = WebPDecodeRGBA(data, size, &width, &height);
  if (!output) {
    throw std::runtime_error("Failed to decode image");
  }

  DecodedImage info;
  info.width = width;
  info.height = height;
  info.rgba.assign(output, output + width * height * 4);
}

std::vector<uint8_t> encodeLosslessRGBA(const uint8_t* rgba, int width, int height, int stride = 0) {
  if (stride == 0) {
    stride = width * 4;
  }

  if (!rgba) {
    throw std::runtime_error("Failed to encode image");
  }

  if (width <= 0 || height <= 0) {
    throw std::runtime_error("Invalid image dimensions");
  }

  if (stride < width * 4) {
    throw std::runtime_error("Invalid stride");
  }

  WebPConfig config;
  if (!WebPConfigInit(&config)) {
    throw std::runtime_error("Failed to initialize WebPConfig");
  }

  config.lossless = 1;
  config.quality = 100;
  config.method = 6;

  if (!WebPValidateConfig(&config)) {
    throw std::runtime_error("Invalid WebPConfig");
  }

  uint8_t* output = nullptr;
  size_t output_size = WebPEncodeLosslessBGRA(rgba, width, height, stride, &output);
  if (output_size == 0 || !output) {
    throw std::runtime_error("Failed to encode image");
  }

  std::vector<uint8_t> result(output, output + output_size);
  WebPFree(output);

  return result;
}

}  // namespace splat