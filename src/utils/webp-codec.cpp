/***********************************************************************************
 *
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
 *
 ***********************************************************************************/

#include <splat/utils/webp-codec.h>
#include <webp/decode.h>
#include <webp/encode.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace splat::webpcodec {

struct WebPFreeDeleter {
  void operator()(void* ptr) const {
    if (ptr != nullptr) {
      WebPFree(ptr);
    }
  }
};

using WebPDataPtr = std::unique_ptr<uint8_t, WebPFreeDeleter>;

std::tuple<std::vector<uint8_t>, int, int> decodeRGBA(const std::vector<uint8_t>& webp) {
  int width = 0;
  int height = 0;
  uint8_t* rgba_buffer = WebPDecodeRGBA(webp.data(), webp.size(), &width, &height);

  if (rgba_buffer == nullptr) {
    throw std::runtime_error("WebP decode failed. Could not decode data.");
  }

  WebPDataPtr decoded_data(rgba_buffer);

  const size_t size = static_cast<size_t>(width) * height * 4;

  std::vector<uint8_t> result_data(decoded_data.get(), decoded_data.get() + size);

  return {result_data, width, height};
}

std::vector<uint8_t> encodeLosslessRGBA(const std::vector<uint8_t>& rgba, int width, int height, int stride) {
  if (stride == 0) {
    stride = width * 4;
  }

  uint8_t* output_buffer = nullptr;
  size_t output_size = 0;

  output_size = WebPEncodeLosslessRGBA(rgba.data(), width, height, stride, &output_buffer);

  if (output_size == 0) {
    throw std::runtime_error("WebP lossless encode failed. Output size is zero.");
  }

  WebPDataPtr encoded_data(output_buffer);

  std::vector<uint8_t> result_data(encoded_data.get(), encoded_data.get() + output_size);

  return result_data;
}

}  // namespace splat::webpcodec
