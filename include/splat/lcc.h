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

#include <Eigen/Dense>
#include <map>

namespace splat {

// lod data in data.bin
struct LccLod {
  int points;     // number of splats
  size_t offset;  // offset
  int size;       // data size
};

// The scene uses a quadtree for spatial partitioning,
// with each unit having its own xy index (starting from 0) and multiple layers of lod data
struct LccUnitInfo {
  int x;                     // x index
  int y;                     // y index
  std::vector<LccLod> lods;  // lods
};

struct CompressInfo {
  Eigen::Vector3d scaleMin;     // min scale
  Eigen::Vector3d scaleMax;     // max scale
  Eigen::Vector3d shMin;        // min sh
  Eigen::Vector3d shMax;        // max sh
  Eigen::Vector3d envScaleMin;  // min environment scale
  Eigen::Vector3d envScaleMax;  // max environment scale
  Eigen::Vector3d envShMin;     // min environment sh
  Eigen::Vector3d envShMax;     // max environment sh
};

// parameters used to convert LCC data into GSplatData
struct LccParam {
  int totalSplats;
  int targetLod;
  CompressInfo compressInfo;
  std::vector<LccUnitInfo> unitInfos;
  std::string dataFile;
  std::string shFile;
};

struct ProcessUnitContext {
  LccUnitInfo info;
  int targetLod;
  std::string dataFile;
  std::string shFile;
  CompressInfo compressInfo;
  double propertyOffset;
  std::map<std::string, std::vector<float>> properties;
  std::vector<float> properties_f_rest;
};

}  // namespace splat
