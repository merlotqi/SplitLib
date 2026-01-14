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

#include <string>
#include <memory>

namespace splat {

class DataTable;

/**
 * @brief Reads and parses a PLY (Polygon File Format) file from disk.
 * 
 * This function loads a PLY file, parses its header and data sections, and returns
 * a DataTable containing the vertex data. The function supports both ASCII and binary
 * PLY formats and handles data in chunks for memory efficiency.
 * 
 * @param[in] filename Path to the PLY file to be read.
 * 
 * @return std::unique_ptr<DataTable> A smart pointer to a DataTable containing
 *         the vertex data from the PLY file. If the file contains compressed data,
 *         it will be decompressed automatically.
 * 
 * @throws std::runtime_error If:
 *         - The file cannot be opened
 *         - The file header is invalid or missing the PLY magic bytes
 *         - The header exceeds the maximum size (128KB)
 *         - The 'end_header' marker is not found
 *         - Data chunks cannot be read properly
 *         - The file does not contain a vertex element
 * 
 * @note The function reads data in chunks of 1024 rows at a time to optimize memory usage.
 *       Non-vertex elements are stored in the PlyData structure but only vertex data is returned.
 *       If the PLY data is compressed, it will be decompressed before returning.
 * 
 * @par File Format Support:
 * - Binary PLY format (little-endian)
 * - Structured elements with properties
 * - Optional compression (automatically detected and decompressed)
 * 
 * @see parseHeader() For header parsing details
 * @see isCompressedPly() For compression detection
 * @see decompressPly() For decompression logic
 */
std::unique_ptr<DataTable> readPly(const std::string& filename);

}  // namespace splat
