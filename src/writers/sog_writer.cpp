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

#include <splat/maths/kmeans.h>
#include <splat/strings.h>
#include <splat/webp-codec.h>
#include <splat/writers/sog_writer.h>
#include <splat/zip_writer.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>

namespace fs = std::filesystem;

namespace splat {

std::vector<std::array<float, 2>> calcMinMax(const DataTable& dataTable, const std::vector<std::string>& columnNames,
                                             const std::vector<uint32_t>& indices) {
  const size_t numCols = columnNames.size();

  std::vector<std::array<float, 2>> minMax(
      numCols, {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()});

  std::vector<const Column*> targetColumns;
  for (const auto& name : columnNames) {
    targetColumns.push_back(&dataTable.getColumnByName(name));
  }

  for (uint32_t idx : indices) {
    for (size_t j = 0; j < numCols; ++j) {
      float value = targetColumns[j]->getValue<float>(idx);

      auto& [currentMin, currentMax] = minMax[j];
      if (value < currentMin) currentMin = value;
      if (value > currentMax) currentMax = value;
    }
  }

  return minMax;
}

static const std::array<std::string, 45> shNames = {"f_rest_0",  "f_rest_1",  "f_rest_2",  "f_rest_3",  "f_rest_4",

                                                    "f_rest_5",  "f_rest_6",  "f_rest_7",  "f_rest_8",  "f_rest_9",

                                                    "f_rest_10", "f_rest_11", "f_rest_12", "f_rest_13", "f_rest_14",

                                                    "f_rest_15", "f_rest_16", "f_rest_17", "f_rest_18", "f_rest_19",

                                                    "f_rest_20", "f_rest_21", "f_rest_22", "f_rest_23", "f_rest_24",

                                                    "f_rest_25", "f_rest_26", "f_rest_27", "f_rest_28", "f_rest_29",

                                                    "f_rest_30", "f_rest_31", "f_rest_32", "f_rest_33", "f_rest_34",

                                                    "f_rest_35", "f_rest_36", "f_rest_37", "f_rest_38", "f_rest_39",

                                                    "f_rest_40", "f_rest_41", "f_rest_42", "f_rest_43", "f_rest_44"};

static float logTransform(float value) { return std::copysign(1.0f, value) * std::logf(std::abs(value) + 1.0f); }

static std::vector<uint32_t> generateIndices(DataTable& dataTable) {
  std::vector<uint32_t> result(dataTable.getNumRows());
  std::iota(result.begin(), result.end(), 0);
  generateOrdering(dataTable, result);
  return result;
}

std::tuple<DataTable, DataTable> cluster1d(const DataTable& dataTable, int iterations) {
  const auto numColumns = dataTable.getNumColumns();
  const auto numRows = dataTable.getNumRows();

  // construct 1d points from the columns of data
  std::vector<float> data(numRows * numColumns, 0.f);
  for (const auto& c : dataTable.columns) {
    auto&& _data = c.as<float>();
    data.insert(data.end(), _data.begin(), _data.end());
  }

  DataTable src({{"name", std::move(data)}});

  auto&& [centroids, labels] = kmeans(src, 256, iterations);

  // order centroids smallest to largest
  std::vector<float> centroidsData = centroids.getColumn(0).as<float>();
  std::vector<size_t> order(centroidsData.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return centroidsData[a] < centroidsData[b]; });

  // reorder centroids
  auto tmp = centroidsData;
  for (size_t i = 0; i < order.size(); i++) {
    centroidsData[i] = tmp[order[i]];
  }

  std::vector<uint32_t> invOrder(order.size());
  for (size_t i = 0; i < order.size(); i++) {
    invOrder[order[i]] = i;
  }

  // reorder labels
  for (size_t i = 0; i < labels.size(); i++) {
    labels[i] = invOrder[labels[i]];
  }

  DataTable result;
  // 
}

void writeSog(const std::string& filename, const DataTable& dataTable, const std::string& outputFilename,
              const Options& options, const std::vector<uint32_t>& indices) {
  const auto isBundle = strings::endsWith(strings::toLowerCase(filename), ".sog");

  std::shared_ptr<ZipWriter> zipWriter;
  if (isBundle) {
    zipWriter = std::make_shared<ZipWriter>(outputFilename);
  }
  const size_t numRows = indices.size();
  const size_t width = ceil(sqrt(numRows) / 4) * 4;
  const size_t height = ceil(numRows / width / 4) * 4;
  const size_t channels = 4;

  // the layout function determines how the data is packed into the output texture.

  auto writeWebp = [&](const std::string& filename, const std::vector<uint8_t>& data, size_t w, size_t h) {
    fs::path outputDir = fs::path(outputFilename).parent_path();
    fs::path pathname = outputDir / filename;
    std::cout << "writing " << pathname << std::endl;

    std::vector<uint8_t> webp = webpCodec::encodeLosslessRGBA(data, w, h);
    if (zipWriter) {
      zipWriter->writeFile(filename, webp);
    } else {
      std::ofstream out(filename, std::ios::binary);
      out.write(reinterpret_cast<const char*>(webp.data()), webp.size());
      out.flush();
      out.close();
    }
  };

  auto writeTableData = [&](const std::string& filename, const DataTable& dataTable, size_t w, size_t h) {};

  auto writeMeans = [&]() -> std::pair<std::vector<float>, std::vector<float>> {
    std::vector<uint8_t> meansL(width * height * channels);
    std::vector<uint8_t> meansU(width * height * channels);
    static std::vector<std::string> meansNames = {"x", "y", "z"};
    auto meansMinMax = calcMinMax(dataTable, meansNames, indices);
    std::vector<int> meansColumnIdxs;
    for (const auto& name : meansNames) {
      meansColumnIdxs.push_back(dataTable.getColumnIndex(name));
    }
    Row row;

    for (size_t i = 0; i < indices.size(); i++) {
      auto process = [&](const float& value, int axisIdx) -> uint16_t {
        float val = logTransform(value);
        float minV = meansMinMax[axisIdx][0];
        float maxV = meansMinMax[axisIdx][1];

        float normalized = (val - minV) / (maxV - minV);
        return static_cast<uint16_t>(std::clamp(normalized * 65535.0f, 0.0f, 65535.0f));
      };

      dataTable.getRow(indices[i], row, meansColumnIdxs);
      uint16_t x = process(row["x"], 0);
      uint16_t y = process(row["y"], 1);
      uint16_t z = process(row["z"], 2);

      meansL[i * 4 + 0] = static_cast<uint8_t>(x & 0xff);
      meansL[i * 4 + 1] = static_cast<uint8_t>(y & 0xff);
      meansL[i * 4 + 2] = static_cast<uint8_t>(z & 0xff);
      meansL[i * 4 + 3] = 0xff;

      meansU[i * 4 + 0] = static_cast<uint8_t>((x >> 8) & 0xff);
      meansU[i * 4 + 1] = static_cast<uint8_t>((y >> 8) & 0xff);
      meansU[i * 4 + 2] = static_cast<uint8_t>((z >> 8) & 0xff);
      meansU[i * 4 + 3] = 0xff;
    }

    writeWebp("means_l.webp", meansL, width, height);
    writeWebp("means_u.webp", meansU, width, height);

    std::vector<float> _mins;
    _mins.reserve(meansMinMax.size());
    std::vector<float> _maxs;
    _maxs.reserve(meansMinMax.size());

    for (const auto& [u, v] : meansMinMax) {
      _mins.push_back(u);
      _maxs.push_back(v);
    }
    return {_mins, _maxs};
  };

  auto writeQuaternions = [&]() {
    std::vector<uint8_t> quats(width * height * channels);
    static std::vector<std::string> quatsNames = {"rot_0", "rot_1", "rot_2", "rot_3"};
    std::vector<int> quatsColumnIdxs;
    for (const auto& name : quatsNames) {
      quatsColumnIdxs.push_back(dataTable.getColumnIndex(name));
    }
    std::array<float, 4> q = {0.0, 0.0, 0.0, 0.0};

    Row row;
    for (size_t i = 0; i < indices.size(); i++) {
      dataTable.getRow(indices[i], row, quatsColumnIdxs);
      q[0] = row["rot_0"];
      q[1] = row["rot_1"];
      q[2] = row["rot_2"];
      q[3] = row["rot_3"];

      const float l = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

      // normalize
      std::for_each(q.begin(), q.end(), [&](float& v) { v /= l; });

      // find max component
      auto it = std::max_element(q.begin(), q.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });
      const size_t maxComp = std::distance(q.begin(), it);

      // invert if max component is negative
      if (q[maxComp] < 0) {
        std::for_each(q.begin(), q.end(), [](float& v) { v = -v; });
      }

      // scale by sqrt(2) to fit in [-1, 1] range
      std::for_each(q.begin(), q.end(), [](float& v) { v *= M_SQRT2; });

      static const int QUAT_IDX_MAP[4][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 1, 2}};

      const int* idx = QUAT_IDX_MAP[maxComp];
      quats[i * 4 + 0] = static_cast<uint8_t>((q[idx[0]] * 0.5f + 0.5f) * 255.0f + 0.5f);
      quats[i * 4 + 1] = static_cast<uint8_t>((q[idx[1]] * 0.5f + 0.5f) * 255.0f + 0.5f);
      quats[i * 4 + 2] = static_cast<uint8_t>((q[idx[2]] * 0.5f + 0.5f) * 255.0f + 0.5f);
      quats[i * 4 + 3] = static_cast<uint8_t>(252 + maxComp);
    }

    writeWebp("quats.webp", quats, width, height);
  };

  auto writeScales = [&]() {
    auto&& [centroids, labels] = cluster1d(dataTable.clone({"scale_0, scale_1, scale_2"}), 10);

    writeTableData("scales.webp", labels, width, height);

    //
  };

  auto writeColors = [&]() {

  };

  auto writeSH = [&](int shBands) {
    static std::array<int, 4> _ = {0, 3, 8, 15};
    const auto shCoeffs = _.at(shBands);
  };
}

}  // namespace splat
