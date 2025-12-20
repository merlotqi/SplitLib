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
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace splat {

using Row = std::map<std::string, float>;

enum class ColumnType {
  INT8,
  UINT8,
  INT16,
  UINT16,
  INT32,
  UINT32,
  FLOAT32,
  FLOAT64,
};

using TypedArray = std::variant<std::vector<int8_t>,    // Int8Array
                                std::vector<uint8_t>,   // Uint8Array
                                std::vector<int16_t>,   // Int16Array
                                std::vector<uint16_t>,  // Uint16Array
                                std::vector<int32_t>,   // Int32Array
                                std::vector<uint32_t>,  // Uint32Array
                                std::vector<float>,     // Float32Array
                                std::vector<double>     // Float64Array
                                >;

template <typename T>
constexpr size_t getTypedArrayIndex() {
  return std::variant<std::vector<int8_t>, std::vector<uint8_t>, std::vector<int16_t>, std::vector<uint16_t>,
                      std::vector<int32_t>, std::vector<uint32_t>, std::vector<float>, std::vector<double>>(
             std::in_place_type<std::vector<T>>)
      .index();
}

struct Column {
  std::string name;
  TypedArray data;

  ColumnType getType() const {
    switch (data.index()) {
      case 0:
        return ColumnType::INT8;
      case 1:
        return ColumnType::UINT8;
      case 2:
        return ColumnType::INT16;
      case 3:
        return ColumnType::UINT16;
      case 4:
        return ColumnType::INT32;
      case 5:
        return ColumnType::UINT32;
      case 6:
        return ColumnType::FLOAT32;
      case 7:
        return ColumnType::FLOAT64;
      default:
        throw std::runtime_error("Unknown TypedArray variant index.");
    }
  }

  size_t length() const {
    return std::visit([](const auto& arg) -> size_t { return arg.size(); }, data);
  }

  template <typename T>
  const std::vector<T>& as() const {
    return std::get<std::vector<T>>(data);
  }

  template <typename T>
  std::vector<T>& as() {
    return std::get<std::vector<T>>(data);
  }

  template <typename T>
  T getValue(size_t index) const {
    if (index >= length()) {
      throw std::out_of_range("Index out of range in getValue.");
    }

    if constexpr (std::is_same_v<T, std::string>) {
      return std::visit(
          [index](const auto& vec) -> std::string {
            using Q = typename std::decay_t<decltype(vec)>::value_type;
            return std::to_string(vec[index]);
          },
          data);

    } else {
      return std::visit(
          [index](const auto& vec) -> T {
            using Q = typename std::decay_t<decltype(vec)>::value_type;
            // Compile-time check: Ensure the internal type is convertible to T
            if constexpr (!std::is_convertible_v<Q, T>) {
              throw std::runtime_error("Internal type cannot be safely converted to requested type T.");
            }
            // Note: Range/precision loss checks for the *internal* value (e.g., int32_t to int8_t)
            // are omitted here, as the focus is on the T return type.
            return static_cast<T>(vec[index]);
          },
          data);
    }
  }

  template <typename T>
  void setValue(size_t index, T value) {
    if (index >= length()) {
      throw std::out_of_range("Column index out of range for setValue.");
    }

    auto visitor = [index, value](auto& vec) {
      using VectorType = std::decay_t<decltype(vec)>;
      using InternalType = typename VectorType::value_type;

      if constexpr (std::is_same_v<T, std::string>) {
        // Handle string input
        const std::string& str_value = value;
        try {
          if constexpr (std::is_same_v<InternalType, int8_t> || std::is_same_v<InternalType, int16_t> ||
                        std::is_same_v<InternalType, int32_t>) {
            // For signed integers
            long long temp = std::stoll(str_value);
            if (temp > static_cast<long long>(std::numeric_limits<InternalType>::max()) ||
                temp < static_cast<long long>(std::numeric_limits<InternalType>::min())) {
              throw std::range_error("String value out of range for signed integer type.");
            }
            vec[index] = static_cast<InternalType>(temp);
          } else if constexpr (std::is_same_v<InternalType, uint8_t> || std::is_same_v<InternalType, uint16_t> ||
                               std::is_same_v<InternalType, uint32_t>) {
            // For unsigned integers
            unsigned long long temp = std::stoull(str_value);
            if (temp > static_cast<unsigned long long>(std::numeric_limits<InternalType>::max())) {
              throw std::range_error("String value out of range for unsigned integer type.");
            }
            vec[index] = static_cast<InternalType>(temp);
          } else if constexpr (std::is_same_v<InternalType, float>) {
            vec[index] = std::stof(str_value);
          } else if constexpr (std::is_same_v<InternalType, double>) {
            vec[index] = std::stod(str_value);
          } else {
            throw std::runtime_error("Unsupported type for string conversion.");
          }
        } catch (const std::invalid_argument&) {
          throw std::runtime_error("Invalid argument for string to number conversion.");
        } catch (const std::out_of_range&) {
          throw std::range_error("String value out of range.");
        }
      } else {
        // Handle non-string input
        using ValueType = std::decay_t<T>;

        if constexpr (!std::is_convertible_v<ValueType, InternalType>) {
          throw std::runtime_error("Input type cannot be converted to internal column type.");
        }

        // Check for potential overflow/truncation
        if constexpr (std::is_integral_v<InternalType>) {
          // Check if value fits in internal type
          if constexpr (std::is_integral_v<ValueType>) {
            // Both are integral types
            if (value > static_cast<ValueType>(std::numeric_limits<InternalType>::max()) ||
                value < static_cast<ValueType>(std::numeric_limits<InternalType>::min())) {
              throw std::range_error("Value exceeds range of internal integer type.");
            }
          } else if constexpr (std::is_floating_point_v<ValueType>) {
            // Floating point to integer conversion
            // Check for overflow
            if (value > static_cast<ValueType>(std::numeric_limits<InternalType>::max()) ||
                value < static_cast<ValueType>(std::numeric_limits<InternalType>::min())) {
              throw std::range_error("Value exceeds range of internal integer type.");
            }
            // Check for truncation
            if (std::abs(value - std::round(value)) > std::numeric_limits<ValueType>::epsilon() * 10) {
              throw std::range_error("Floating-point value cannot be exactly represented in internal integer type.");
            }
          }
        } else if constexpr (std::is_floating_point_v<InternalType>) {
          // Floating point to floating point conversion
          if constexpr (std::is_same_v<InternalType, float> && std::is_same_v<ValueType, double>) {
            // double to float conversion - check for overflow
            if (std::abs(value) > std::numeric_limits<float>::max()) {
              throw std::range_error("Double value exceeds float range.");
            }
          }
        }

        // Safe to assign
        vec[index] = static_cast<InternalType>(value);
      }
    };

    std::visit(visitor, data);
  }

  float getValue(size_t index) const { return getValue<float>(index); }

  void setValue(size_t index, float value) { setValue<float>(index, value); }

  size_t bytePreElement() const {
    return std::visit(
        [](const auto& vec) -> size_t {
          using T = typename std::decay_t<decltype(vec)>::value_type;
          return sizeof(T);
        },
        data);
  }

  size_t totalByteSize() const { return length() * bytePreElement(); }

  const uint8_t* rawPointer() const {
    return std::visit([](const auto& vec) -> const uint8_t* { return reinterpret_cast<const uint8_t*>(vec.data()); },
                      data);
  }

  uint8_t* rawPointer() {
    return std::visit([](auto& vec) -> uint8_t* { return reinterpret_cast<uint8_t*>(vec.data()); }, data);
  }

  void fillBuffer(float* dest) const {
    std::visit(
        [dest](const auto& vec) {
          for (size_t i = 0; i < vec.size(); i++) {
            dest[i] = static_cast<float>(vec[i]);
          }
        },
        data);
  }
};

class DataTable {
 public:
  std::vector<Column> columns;

  DataTable() = default;
  DataTable(const std::vector<Column>& columns);

  DataTable(const DataTable& other) = default;
  DataTable& operator=(const DataTable& other) = default;
  DataTable(DataTable&& other) = default;
  DataTable& operator=(DataTable&& other) = default;

  size_t getNumRows() const;
  Row getRow(size_t index, const std::vector<int>& columnIdx = {}) const;
  void getRow(size_t index, Row& row, const std::vector<int>& columnIdx = {}) const;
  void setRow(size_t index, const Row& row);
  size_t getNumColumns() const;

  std::vector<std::string> getColumnNames() const;
  std::vector<ColumnType> getColumnTypes() const;
  const Column& getColumn(size_t index) const;
  Column& getColumn(size_t index);
  int getColumnIndex(const std::string& name) const;
  const Column& getColumnByName(const std::string& name) const;
  Column& getColumnByName(const std::string& name);

  bool hasColumn(const std::string& name) const;
  void addColumn(const Column& column);
  bool removeColumn(const std::string& name);

  DataTable clone(const std::vector<std::string>& columnNames = {}) const;
  DataTable permuteRows(const std::vector<uint32_t>& indices) const;

  template <typename T>
  const std::vector<T>& getRawColumnData(const std::string& name) const {
    const Column& col = getColumnByName(name);

    try {
      return std::get<std::vector<T>>(col.data);
    } catch (const std::bad_variant_access&) {
      throw std::runtime_error(
          "Column type mismatch for '" + name + "'. Expected type index " +
          std::to_string(
              std::variant<std::vector<int8_t>, std::vector<uint8_t>, std::vector<int16_t>, std::vector<uint16_t>,
                           std::vector<int32_t>, std::vector<uint32_t>, std::vector<float>, std::vector<double>>(
                  std::in_place_type<std::vector<T>>)
                  .index()) +
          ", got index " + std::to_string(col.data.index()));
    } catch (...) {
      throw std::runtime_error("Unknown error while getting raw column data for '" + name + "'.");
    }
  }

  template <typename T>
  const std::vector<T>& getRawColumnData(size_t index) const {
    const Column& col = getColumn(index);

    try {
      return std::get<std::vector<T>>(col.data);
    } catch (const std::bad_variant_access&) {
      throw std::runtime_error(
          "Column type mismatch for '" + col.name + "'. Expected type index " +
          std::to_string(
              std::variant<std::vector<int8_t>, std::vector<uint8_t>, std::vector<int16_t>, std::vector<uint16_t>,
                           std::vector<int32_t>, std::vector<uint32_t>, std::vector<float>, std::vector<double>>(
                  std::in_place_type<std::vector<T>>)
                  .index()) +
          ", got index " + std::to_string(col.data.index()));
    } catch (...) {
      throw std::runtime_error("Unknown error while getting raw column data for '" + col.name + "'.");
    }
  }
};

std::vector<uint32_t>& generateOrdering(DataTable& dataTable, std::vector<uint32_t>& indices);

void transform(DataTable& dataTable, const Eigen::Vector3f& t, const Eigen::Quaternionf& r, float s);

}  // namespace splat

template const std::vector<int8_t>& splat::DataTable::getRawColumnData<int8_t>(size_t) const;
template const std::vector<uint8_t>& splat::DataTable::getRawColumnData<uint8_t>(size_t) const;
template const std::vector<int16_t>& splat::DataTable::getRawColumnData<int16_t>(size_t) const;
template const std::vector<uint16_t>& splat::DataTable::getRawColumnData<uint16_t>(size_t) const;
template const std::vector<int32_t>& splat::DataTable::getRawColumnData<int32_t>(size_t) const;
template const std::vector<uint32_t>& splat::DataTable::getRawColumnData<uint32_t>(size_t) const;
template const std::vector<float>& splat::DataTable::getRawColumnData<float>(size_t) const;
template const std::vector<double>& splat::DataTable::getRawColumnData<double>(size_t) const;

template const std::vector<int8_t>& splat::DataTable::getRawColumnData<int8_t>(const std::string&) const;
template const std::vector<uint8_t>& splat::DataTable::getRawColumnData<uint8_t>(const std::string&) const;
template const std::vector<int16_t>& splat::DataTable::getRawColumnData<int16_t>(const std::string&) const;
template const std::vector<uint16_t>& splat::DataTable::getRawColumnData<uint16_t>(const std::string&) const;
template const std::vector<int32_t>& splat::DataTable::getRawColumnData<int32_t>(const std::string&) const;
template const std::vector<uint32_t>& splat::DataTable::getRawColumnData<uint32_t>(const std::string&) const;
template const std::vector<float>& splat::DataTable::getRawColumnData<float>(const std::string&) const;
template const std::vector<double>& splat::DataTable::getRawColumnData<double>(const std::string&) const;
