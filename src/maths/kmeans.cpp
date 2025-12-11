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

#include <numeric>
#include <random>
#include <set>

namespace splat {

static void initializeCentroids(const DataTable& dataTable, DataTable& centroids) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> dis(0, dataTable.getNumRows() - 1);

  std::set<size_t> chosenRows;
  for (size_t i = 0; i < centroids.getNumRows(); ++i) {
    size_t candidateRow = 0;
    do {
      candidateRow = dis(gen);
    } while (chosenRows.count(candidateRow));

    chosenRows.insert(candidateRow);
    Row row = dataTable.getRow(candidateRow);
    centroids.setRow(i, row);
  }
}

static void initializeCentroids1D(const DataTable& dataTable, DataTable& centroids) {
  auto m = std::numeric_limits<float>::infinity();
  auto n = -std::numeric_limits<float>::infinity();

  auto col0 = dataTable.getColumn(0);
  for (size_t i = 0; i < dataTable.getNumRows(); ++i) {
    const auto value = col0.getValue<float>(i);
    if (value < m) m = value;
    if (value > n) n = value;
  }

  auto& centroidsCol0 = centroids.getColumn(0);
  for (size_t i = 0; i < centroids.getNumRows(); ++i) {
    centroidsCol0.setValue<float>(i, m + (n - m) * i / (centroids.getNumRows() - 1));
  }
}

static void calcAverage(const DataTable& dataTable, const std::vector<int>& cluster,
                        std::map<std::string, float>& row) {
  const auto keys = dataTable.getColumnNames();

  for (size_t i = 0; i < keys.size(); ++i) {
    row[keys[i]] = 0.f;
  }

  for (size_t i = 0; i < cluster.size(); ++i) {
    auto dataRow = dataTable.getRow(cluster[i]);

    for (size_t j = 0; j < keys.size(); ++j) {
      const auto key = keys[i];
      row[key] += dataRow[key];
    }
  }

  if (cluster.size() > 0) {
    for (size_t i = 0; i < keys.size(); ++i) {
      row[keys[i]] /= cluster.size();
    }
  }
}

static void clusterKdTreeCpu(const DataTable& points, DataTable &centroids, std::vector<uint32_t>& labels) {

}

std::pair<DataTable, std::vector<int>> kmeans(const DataTable& points, int k, int iterations) {
  // too few data points
  if (points.getNumRows() < k) {
    std::vector<int> labels(points.getNumRows(), 0);
    std::iota(labels.begin(), labels.end(), 0);
    return {points.clone(), labels};
  }

  DataTable centroids;
  for (auto& c : points.columns) {
    centroids.addColumn({c.name, std::vector<float>(k, 0)});
  }

  if (points.getNumColumns() == 1) {
    initializeCentroids1D(points, centroids);
  } else {
    initializeCentroids(points, centroids);
  }
}
}  // namespace splat
