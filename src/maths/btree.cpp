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

#include <numeric>

namespace splat {

/**
 * @brief Partitions the index array (idx) around the k-th smallest element based on the values in 'data'.
 * Implements the Quickselect algorithm with Median-of-Three pivot selection.
 *
 * @param data The data array whose values are used for comparison.
 * @param idx The index array to be partitioned (in-place modification).
 * @param k The index (0-based) of the element to select (e.g., k=indices.size()/2 for median).
 * @return The value of the k-th element in the original index array (before partitioning).
 */
uint32_t quickselect(const std::vector<double>& data, std::vector<uint32_t>& idx, size_t k) {
  // Utility functions to handle value access and index swapping
  auto valAt = [&](size_t p) -> double {
    // idx[p] contains the original index, which is used to look up the actual value in data.
    return data[idx[p]];
  };

  auto swap = [&](size_t i, size_t j) {
    if (i == j) return;
    uint32_t t = idx[i];
    idx[i] = idx[j];
    idx[j] = t;
  };

  size_t n = idx.size();
  if (k >= n) {
    throw std::out_of_range("k is out of range for quickselect.");
  }

  size_t l = 0;
  size_t r = n - 1;

  // Use a non-recursive loop
  while (true) {
    if (r <= l + 1) {
      // Base case: 1 or 2 elements left
      if (r == l + 1 && valAt(r) < valAt(l)) {
        swap(l, r);
      }
      // k is in the current [l, r] range (which is size 1 or 2)
      if (k >= l && k <= r) {
        return idx[k];
      } else {
        // Should not happen if k was originally in [0, n-1]
        throw std::runtime_error("Internal error in quickselect partition bounds.");
      }
    }

    // Median-of-three pivot selection (using values via idx)
    size_t mid = l + (r - l) / 2;  // Equivalent to (l + r) >>> 1 in JS

    // 1. Ensure l < mid < r for the comparison set
    // The pivot element will be stored at l+1 after this block.
    swap(mid, l + 1);

    // 2. Sort the three indices {l, l+1, r} based on their values in `data`
    // Put smallest at l, median at l+1, largest at r
    if (valAt(l) > valAt(r)) swap(l, r);
    if (valAt(l + 1) > valAt(r)) swap(l + 1, r);
    if (valAt(l) > valAt(l + 1)) swap(l, l + 1);

    size_t i = l + 1;
    size_t j = r;

    // Pivot is now at l+1, use its value for partitioning
    float pivotVal = valAt(l + 1);
    uint32_t pivotIdx = idx[l + 1];

    // 3. Partition [l+2, r-1] around the pivot value
    while (true) {
      // Find element greater than pivot from left
      do {
        i++;
      } while (i <= r && valAt(i) < pivotVal);

      // Find element less than pivot from right
      do {
        j--;
      } while (j >= l && valAt(j) > pivotVal);

      if (j < i) break;
      swap(i, j);
    }

    // 4. Place the pivot (originally at l+1) in its final sorted position (j)
    // Swap the value at l+1 (median index) with the value at j (partition boundary)
    idx[l + 1] = idx[j];
    idx[j] = pivotIdx;  // Put pivot index into final position

    // 5. Narrow down the search range
    if (j == k) {
      return idx[k];
    } else if (j > k) {
      r = j - 1;  // Search in the left sub-array
    } else {      // j < k
      l = j + 1;  // Search in the right sub-array (l = i is from original, but j+1 is clearer)
                  // Note: The original JS code used `l = i;`. Since `i` is the first element
                  // strictly greater than or equal to the pivot, `l = j + 1` should be safer
                  // unless the implementation relies on the specific behavior of `i`.
                  // Let's stick to the standard:
                  // l = j + 1;
    }
  }
}
AABB::AABB(const std::vector<double>& min, const std::vector<double>& max) : min(min), max(max) {}

/**
 * @brief Calculates the index (0-based) of the largest axis (dimension) of the AABB.
 * @return The index of the largest axis, or -1 if min/max vectors are empty.
 */
int AABB::largestAxis() const {
  const size_t length = min.size();
  if (length == 0) return -1;
  auto l = -std::numeric_limits<double>::infinity();
  int result = -1;
  for (size_t i = 0; i < length; i++) {
    const double e = max[i] - min[i];
    if (e > l) {
      l = e;
      result = static_cast<int>(i);
    }
  }
  return result;
}

/**
 * @brief Calculates the length of the AABB along its largest dimension.
 * @return The largest dimension size, or 0 if empty.
 */
double AABB::largestDim() const {
  const double a = largestAxis();
  return max[a] - min[a];
}

/**
 * @brief Computes the AABB that tightly encloses the centroids specified by the indices.
 * Resizes min/max vectors if necessary.
 * @param centroids The DataTable containing the centroid coordinate columns.
 * @param indices The indices of the rows to include in the AABB calculation.
 * @return A reference to the updated Aabb object.
 */
AABB& AABB::fromCentroids(const DataTable& centroids, const std::vector<uint32_t>& indices) {
  for (size_t i = 0; i < centroids.getNumColumns(); i++) {
    const auto data = centroids.getColumn(i);
    double m = std::numeric_limits<double>::infinity();
    double n = -std::numeric_limits<double>::infinity();

    for (auto index : indices) {
      const double v = data.getValue(index);
      m = std::min(v, m);
      n = std::max(v, n);
    }
    min[i] = m;
    max[i] = n;
  }
  return *this;
}

// Leaf size threshold
static constexpr size_t LEAF_SIZE_THRESHOLD = 256;

BTree::BTree(const DataTable& centroids) : centroids(centroids) {
  const size_t numRows = centroids.getNumRows();

  // 1. Initialize the index array (0, 1, 2, ..., numRows-1)
  std::vector<uint32_t> indices(numRows);
  std::iota(indices.begin(), indices.end(), 0);

  // 2. Build the tree recursively
  root = recurse(std::move(indices));
}

/**
 * @brief Recursive function to build the BTree node.
 * The input vector 'indices' is modified in place and partitioned.
 *
 * @param indices The index vector for the current node's elements (passed by move).
 * @return A unique pointer to the newly created BTreeNode.
 */
std::unique_ptr<BTreeNode> BTree::recurse(std::vector<uint32_t> indices) {
  const size_t currentCount = indices.size();

  // Create the node structure
  auto node = std::make_unique<BTreeNode>();
  node->count = currentCount;

  // 1. Calculate AABB for the current set of indices
  node->aabb.fromCentroids(centroids, indices);

  // 2. Base Case: Leaf Node (indices.length <= 256)
  if (currentCount <= LEAF_SIZE_THRESHOLD) {
    // Store the indices in the leaf node
    node->indices = indices;
    return node;
  }

  // 3. Internal Node: Determine split axis and median

  const int col = node->aabb.largestAxis();
  if (col == -1) {
    // Should not happen if data is valid, but handle case where AABB is flat/empty
    node->indices = indices;
    return node;
  }

  const size_t mid = currentCount / 2;

  // Get the data array for the splitting dimension
  // Assumes column data structure is consistent (e.g., floatData for X, Y, Z)
  const auto& values = centroids.getRawColumnData<double>(col);

  // Partition the 'indices' vector in place around the median (k = mid)
  // This sorts the indices based on the values in 'values'
  // quickselect returns the median value's original index, which is not strictly needed here.
  quickselect(values, indices, mid);

  // 4. Recursive Calls and Sub-array split

  // Create two new vectors to pass to the recursive calls.
  // This involves a copy, which is the necessary cost of partitioning a single vector
  // into two separate, non-contiguous sub-arrays for the children nodes.
  std::vector<uint32_t> leftIndices(indices.begin(), indices.begin() + mid);
  std::vector<uint32_t> rightIndices(indices.begin() + mid, indices.end());

  // Recurse using move semantics to transfer ownership of the index vector
  node->left = recurse(leftIndices);
  node->right = recurse(leftIndices);

  // Ensure the current (parent) node doesn't hold indices
  // indices vector is now empty (moved into rightIndices/leftIndices)
  return node;
}

}  // namespace splat
