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

#pragma once

#include <algorithm>
#include <filesystem>
#include <string>

#include "../utils/data_table.h"

namespace fs = std::filesystem;

namespace reader {
namespace splat {

void read_splat(const fs::path& filepath) {
  const size_t fileSize = fs::file_size(filepath);
  constexpr int BYTES_PER_SPLAT = 32;
  if (fileSize % BYTES_PER_SPLAT != 0) {
    throw std::runtime_error(
        "Invalid .splat file: file size is not a multiple of 32 bytes");
  }

  const size_t numSplat = fileSize / BYTES_PER_SPLAT;
  if (numSplat == 0) {
    throw std::runtime_error("Invalid .splat file: file is empty");
  }

  // Create columns for the standard Gaussian splat data
  const std::vector<Column> columns = {
      // position
      {"x", std::vector<float>(numSplat)},
      {"y", std::vector<float>(numSplat)},
      {"z", std::vector<float>(numSplat)},

      // Scale (stored as linear in .splat, convert to log for internal use)
      {"scale_0", std::vector<float>(numSplat)},
      {"scale_1", std::vector<float>(numSplat)},
      {"scale_2", std::vector<float>(numSplat)},

      // Color/opacity
      {"f_dc_0", std::vector<float>(numSplat)},  // red
      {"f_dc_1", std::vector<float>(numSplat)},  // green
      {"f_dc_2", std::vector<float>(numSplat)},  // blue
      {"opacity", std::vector<float>(numSplat)},

      // Rotation quaternion
      {"rot_0", std::vector<float>(numSplat)},
      {"rot_1", std::vector<float>(numSplat)},
      {"rot_2", std::vector<float>(numSplat)},
      {"rot_3", std::vector<float>(numSplat)},
  };

  // Read data in chunks
  constexpr size_t ChunkSize = 1024;
  const int numChunks = ceil(numSplat / ChunkSize);
  const char* chunkData = (const char*)malloc(ChunkSize * BYTES_PER_SPLAT);

  for (size_t c = 0; c < ChunkSize; ++c) {
    const int numRows = std::min(ChunkSize, numSplat - c * ChunkSize);
    const int bytesToRead = numRows * BYTES_PER_SPLAT;
  }
}

}  // namespace splat
}  // namespace reader
