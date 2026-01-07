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

#include <splat/models/data-table.h>

namespace splat {

/**
 * @brief Sort Gaussian splats in Morton order (Z-order curve) for memory locality
 * 
 * This function computes a Morton ordering (Z-order curve) of the Gaussian splats
 * based on their 3D positions to improve cache locality and memory access patterns.
 * The Morton code interleaves the bits of the 3D coordinates to create a 1D ordering
 * that preserves spatial proximity in multi-dimensional space.
 * 
 * @param dataTable Pointer to the DataTable containing Gaussian splat data. 
 *                  Expected to have at least 'x', 'y', and 'z' columns representing
 *                  the 3D positions of the splats.
 * @param indices Output span that will be filled with the sorted indices in Morton order.
 *                Must be pre-allocated with size equal to the number of rows in dataTable.
 *                On output, indices[i] contains the original index of the i-th splat 
 *                in Morton order.
 * 
 * @note The function assumes the position coordinates are normalized or bounded.
 * @note Morton ordering is particularly effective for octree-based rendering and 
 *       improves performance for spatially coherent operations.
 * @note This is a key optimization for real-time Gaussian splatting rendering pipelines.
 */
void sortMortonOrder(const DataTable* dataTable, absl::Span<uint32_t> indices);

}  // namespace splat
