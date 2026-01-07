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
 * @struct PlyProperty
 * @brief Represents a single property definition in a PLY element
 *
 * Describes a property within a PLY file element, including its name,
 * data type as string, and the corresponding ColumnType for internal use.
 */
struct PlyProperty {
  std::string name;     ///< Property name (e.g., 'x', 'f_dc_0', 'opacity')
  std::string type;     ///< PLY type string (e.g., 'float', 'uchar', 'int32')
  ColumnType dataType;  ///< Internal ColumnType representation
};

/**
 * @struct PlyElement
 * @brief Represents a PLY element (group of properties) in the header
 *
 * Defines an element in a PLY file, such as 'vertex' or 'face', containing
 * multiple properties and the count of items of this element type.
 */
struct PlyElement {
  std::string name;                     ///< Element name (e.g., 'vertex', 'face')
  size_t count;                         ///< Number of items of this element type
  std::vector<PlyProperty> properties;  ///< List of properties in this element
};

/**
 * @struct PlyHeader
 * @brief Complete PLY file header information
 *
 * Contains all metadata from a PLY file header, including comments
 * and element definitions.
 */
struct PlyHeader {
  std::vector<std::string> comments;  ///< Comment lines from the PLY header
  std::vector<PlyElement> elements;   ///< Element definitions in the PLY file
};

/**
 * @struct PlyElementData
 * @brief Contains actual data for a PLY element along with its name
 *
 * Associates a DataTable containing the actual property values with
 * the corresponding element name from the PLY file.
 */
struct PlyElementData {
  std::string name;                      ///< Element name (must match PlyElement::name)
  std::unique_ptr<DataTable> dataTable;  ///< Data table containing property values
};

/**
 * @struct PlyData
 * @brief Complete in-memory representation of PLY file data
 *
 * Contains both the original comments and all element data from
 * a parsed PLY file. This is the primary structure for working with
 * PLY data in memory.
 */
struct PlyData {
  std::vector<std::string> comments;     ///< Comments from the PLY file
  std::vector<PlyElementData> elements;  ///< Data for each element in the PLY file
};

}  // namespace splat
