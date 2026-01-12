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

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace splat {

class DataTable;

class Octree {
 public:
  struct AABB {
    std::array<float, 3> min;
    std::array<float, 3> max;

    AABB() = default;
    AABB(float minX, float minY, float minZ, float maxX, float maxY, float maxZ);
    void getCenter(float&x, float&y, float&z) const;
    void contains(float x, float y, float z);
  };

  struct OctreeNode {
    AABB aabb;
    std::vector<size_t> pointIndices;
    std::array<std::unique_ptr<OctreeNode>, 8> children;
    bool isLeaf = true;
    int depth = 0;
  };

 public:
  explicit Octree(DataTable *table, size_t maxPoints = 32, int maxDepth = 8);

  std::unique_ptr<OctreeNode> root;

 private:
  void build();

  DataTable *dataTable_;
  int maxDepth_;
  int maxPointPerNodes_;
};

}  // namespace splat
