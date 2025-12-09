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

#include <filesystem>
#include <fstream>

#include "../utils/data_table.h"

namespace fs = std::filesystem;

namespace writer {
namespace csv {

std::string stringListJoin(const std::vector<std::string>& strings, const std::string& delimiter) {
  std::string result;
  for (size_t i = 0; i < strings.size(); ++i) {
    result += strings[i];
    if (i < strings.size() - 1) {
      result += delimiter;
    }
  }
  return result;
}

void writeCSV(const fs::path& path, const DataTable& data_table) {
  const size_t len = data_table.row_size();

  std::ofstream file;
  file.open(path);
  file << stringListJoin(data_table.get_column_names(), ",") << std::endl;

  for (size_t i = 0; i < len; ++i) {
    std::string row;
    for (size_t c = 0; c < data_table.columns.size(); c++) {
      if (c) {
        row += ",";
      }
      row += data_table.value_at<std::string>(i, c);
    }
    file << row << std::endl;
  }
}

}  // namespace csv
}  // namespace writer
