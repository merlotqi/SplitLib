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

#include <math.h>
#include <iostream>
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "../utils/zip_reader.h"

namespace reader {
namespace sog {

static std::array<std::vector<uint16_t>, 3> decodeMeans(const std::vector<uint8_t>& lo, const std::vector<uint8_t>& hi,
                                                        size_t count) {
  std::vector<uint16_t> xs(count, 0);
  for (size_t i = 0; i < count; i++) {
    const auto o = i * 4;
    xs[i] = static_cast<uint16_t>(lo[o + 0] | (hi[o + 1] << 8));
  }
  std::vector<uint16_t> ys(count, 0);
  for (size_t i = 0; i < count; i++) {
    const auto o = i * 4;
    ys[i] = static_cast<uint16_t>(lo[o + 2] | (hi[o + 3] << 8));
  }
  std::vector<uint16_t> zs(count, 0);
  for (size_t i = 0; i < count; i++) {
    const auto o = i * 4;
    zs[i] = static_cast<uint16_t>(lo[o + 4] | (hi[o + 5] << 8));
  }
  return {xs, ys, zs};
}

static double invLogTransform(double v) {
  const double a = abs(v);
  const double e = exp(a) - 1;
  return v < 0 ? -e : e;
}

static inline std::array<float, 4> unpackQuat(uint8_t px, uint8_t py, uint8_t pz, uint8_t tag) {
  const uint8_t maxComp = tag - 252;
  const float a = static_cast<float>(px) / 255.0f * 2.0f - 1.0f;
  const float b = static_cast<float>(py) / 255.0f * 2.0f - 1.0f;
  const float c = static_cast<float>(pz) / 255.0f * 2.0f - 1.0f;
  constexpr float sqrt2 = 1.41421356237f;
  std::array<float, 4> comps = {0.0f, 0.0f, 0.0f, 0.0f};
  static constexpr std::array<std::array<uint8_t, 3>, 4> idx = {{{{1, 2, 3}}, {{0, 2, 3}}, {{0, 1, 3}}, {{0, 1, 2}}}};

  const auto& indices = idx[maxComp];
  comps[indices[0]] = a / sqrt2;
  comps[indices[1]] = b / sqrt2;
  comps[indices[2]] = c / sqrt2;

  float t = 1.0f - (comps[0] * comps[0] + comps[1] * comps[1] + comps[2] * comps[2] + comps[3] * comps[3]);
  comps[maxComp] = sqrt(std::max(0.0f, t));

  return comps;
}

static inline double sigmoidInv(double y) {
  const double e = std::min(1 - 1e-6, std::max(1e-6, y));
  return log(e / (1 - e));
}

static std::string toLowerCase(const std::string& str) {
  std::string lower = str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  return lower;
}

static bool end_withs(const std::string& str, const std::string& suffix) {
  return (str.size() >= suffix.size()) && (str.rfind(suffix, str.size() - suffix.size()) != std::string::npos);
}

void read_sog(std::filesystem::path file, const std::string& sourceName) {
  std::map<std::string, std::vector<uint8_t>> entries;
  const std::string lowerName = toLowerCase(sourceName);
  if (end_withs(lowerName, ".sog")) {
    ZipReader zr(file.string());
    const auto list = zr.list();
    for (const auto& e : list) {
      entries.insert({e.name, e.readData()});
    }
  }

  const auto metaBytes = entries["meta.json"];

}

}  // namespace sog
}  // namespace reader
