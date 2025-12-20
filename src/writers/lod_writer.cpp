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

#include <splat/maths/btree.h>
#include <splat/writers/lod_writer.h>
#include <splat/writers/sog_writer.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

namespace fs = std::filesystem;

namespace splat {

struct Aabb {
  Eigen::Vector3f min;
  Eigen::Vector3f max;
};

struct MetaLod {
  size_t file;
  size_t offset;
  size_t count;
};

struct MetaNode {
  Aabb bound;
  std::vector<MetaLod> lods;  // optional
  std::map<int, MetaLod> lods;
};

struct LodMeta {
  size_t lodLevels;
  std::string environment;
  std::vector<std::string> filenames;
  MetaNode tree;
};

static void boundUnion(Aabb& result, const Aabb& a, const Aabb& b) {
  const auto am = a.min;
  const auto aM = a.max;
  const auto bm = b.min;
  const auto bM = b.max;

  auto& rm = result.min;
  auto& rM = result.max;

  rm[0] = std::min(am[0], bm[0]);
  rm[1] = std::min(am[1], bm[1]);
  rm[2] = std::min(am[2], bm[2]);

  rM[0] = std::max(aM[0], bM[0]);
  rM[1] = std::max(aM[1], bM[1]);
  rM[2] = std::max(aM[2], bM[2]);
}

static Aabb calcBound(const DataTable& dataTable, const std::vector<size_t>& indices) {
  // 1. Get references to columns to avoid massive memory copying
  // Ensure .as<float>() returns a const reference: const std::vector<float>&
  const auto& x = dataTable.getColumnByName("x").as<float>();
  const auto& y = dataTable.getColumnByName("y").as<float>();
  const auto& z = dataTable.getColumnByName("z").as<float>();
  const auto& rx = dataTable.getColumnByName("rot_1").as<float>();
  const auto& ry = dataTable.getColumnByName("rot_2").as<float>();
  const auto& rz = dataTable.getColumnByName("rot_3").as<float>();
  const auto& rw = dataTable.getColumnByName("rot_0").as<float>();
  const auto& sx = dataTable.getColumnByName("scale_0").as<float>();
  const auto& sy = dataTable.getColumnByName("scale_1").as<float>();
  const auto& sz = dataTable.getColumnByName("scale_2").as<float>();

  // Initialize overall bounding box with infinity
  Eigen::Vector3f overallMin(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity());
  Eigen::Vector3f overallMax(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity());

  for (size_t index : indices) {
    // a. Extract Position
    Eigen::Vector3f pos(x[index], y[index], z[index]);

    // b. Extract and Normalize Rotation
    // Eigen::Quaternionf constructor order is (w, x, y, z)
    Eigen::Quaternionf q(rw[index], rx[index], ry[index], rz[index]);
    q.normalize();

    // c. Extract Scale (Reversing log-transform: stored scale is ln(s))
    float s0 = std::exp(sx[index]);
    float s1 = std::exp(sy[index]);
    float s2 = std::exp(sz[index]);

    // d. Calculate the World-Space AABB of the Oriented Ellipsoid
    // A Gaussian Splat is effectively an oriented ellipsoid.
    // Instead of transforming 8 corners, we use the property:
    // For a transformation matrix M = R * S, the AABB half-extents H' are:
    // H'_i = sum_j |M_ij| * local_H_j. Since local_H is (1,1,1), H'_i = sum_j |M_ij|.
    Eigen::Matrix3f rotMat = q.toRotationMatrix();

    Eigen::Vector3f halfExtents;
    for (int i = 0; i < 3; ++i) {
      halfExtents[i] = std::abs(rotMat(i, 0) * s0) + std::abs(rotMat(i, 1) * s1) + std::abs(rotMat(i, 2) * s2);
    }

    Eigen::Vector3f currentMin = pos - halfExtents;
    Eigen::Vector3f currentMax = pos + halfExtents;

    // e. Validation Check (Skip NaNs or Infs)
    if (!currentMin.array().isFinite().all() || !currentMax.array().isFinite().all()) {
      continue;
    }

    // f. Expand global AABB
    overallMin = overallMin.cwiseMin(currentMin);
    overallMax = overallMax.cwiseMax(currentMax);
  }

  return {overallMin, overallMax};
}

static std::map<int, int> binIndices() {
  auto recurse = [&]() {

  };

  recurse();
}

void writeLod(const std::string& filename, const DataTable& dataTable, DataTable* envDataTable,
              const std::string& outputFilename, Options options) {
  fs::path outputDir = fs::path(outputFilename).parent_path();

  // ensure top-level output folder exists
  fs::create_directories(outputDir);
  // write the environment sog
  if (envDataTable && envDataTable->getNumRows() > 0) {
    fs::path pathname = outputDir / "env" / "meta.json";
    fs::create_directories(pathname.parent_path());
    std::cout << "writing " << pathname.string() << "..." << std::endl;
    writeSog(pathname.string(), *envDataTable, pathname.string(), options, {});
  }

 // construct a kd-tree based on centroids from all lods
 auto centroidsTable = dataTable.clone({"x", "y", "z"});
}

}  // namespace splat
