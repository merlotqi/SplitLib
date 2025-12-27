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

#include <iostream>
#include <splat/gpu/gpu-clustering.cuh>
#include <cfloat>
#include <cuda_runtime.h>

namespace splat {

#define WORKGROUP_SIZE 64
#define WORKGROUPS_PER_BATCH 1024
#define BATCH_SIZE (WORKGROUPS_PER_BATCH * WORKGROUP_SIZE)
#define CHUNK_SIZE 128

#define CUDA_CHECK(call)                                                                                             \
  do {                                                                                                               \
    cudaError_t err = call;                                                                                          \
    if (err != cudaSuccess) {                                                                                        \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
      exit(EXIT_FAILURE);                                                                                            \
    }                                                                                                                \
  } while (0)

struct Uniforms {
  uint32_t num_points;
  uint32_t num_centroids;
  uint32_t num_columns;
};

uint32_t round_up(uint32_t value, uint32_t multiple) { return ((value + multiple - 1) / multiple) * multiple; }

__global__ void cluster_kernel_flat_column_major(const float* points, const float* centroids, uint32_t* results,
                                                 const Uniforms uniforms) {
  extern __shared__ float shared_chunk[];

  const uint32_t local_id = threadIdx.x;
  const uint32_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t num_columns = uniforms.num_columns;
  const uint32_t num_points = uniforms.num_points;
  const uint32_t num_centroids = uniforms.num_centroids;

  float point[10];
  if (global_id < num_points) {
    for (uint32_t col = 0; col < num_columns; col++) {
      point[col] = points[col * num_points + global_id];
    }
  }

  float mind = FLT_MAX;
  uint32_t mini = 0;

  uint32_t num_chunks = (num_centroids + CHUNK_SIZE - 1) / CHUNK_SIZE;

  for (uint32_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
    uint32_t rows_per_thread = CHUNK_SIZE / WORKGROUP_SIZE;
    uint32_t chunk_start_row = chunk_idx * CHUNK_SIZE;

    uint32_t src_start_row = chunk_start_row + local_id * rows_per_thread;
    uint32_t num_rows_to_copy = min(rows_per_thread, num_centroids - src_start_row);

    if (num_rows_to_copy > 0) {
      uint32_t dst_offset = local_id * rows_per_thread * num_columns;

      for (uint32_t row = 0; row < num_rows_to_copy; row++) {
        uint32_t src_row = src_start_row + row;
        uint32_t dst_row_offset = dst_offset + row * num_columns;

        for (uint32_t col = 0; col < num_columns; col++) {
          shared_chunk[dst_row_offset + col] = centroids[col * num_centroids + src_row];
        }
      }
    }

    __syncthreads();

    if (global_id < num_points) {
      uint32_t chunk_size = min(CHUNK_SIZE, num_centroids - chunk_start_row);

      for (uint32_t c = 0; c < chunk_size; c++) {
        float distance_sq = 0.0f;
        uint32_t centroid_offset = c * num_columns;

        for (uint32_t col = 0; col < num_columns; col++) {
          float diff = point[col] - shared_chunk[centroid_offset + col];
          distance_sq += diff * diff;
        }

        if (distance_sq < mind) {
          mind = distance_sq;
          mini = chunk_start_row + c;
        }
      }
    }

    __syncthreads();
  }

  if (global_id < num_points) {
    results[global_id] = mini;
  }
}

void gpu_cluster_3d_execute(const std::vector<float>& h_points_flat, const std::vector<float>& h_centroids_flat,
                            std::vector<uint32_t>& h_labels) {
  if (h_points_flat.empty() || h_centroids_flat.empty()) {
    std::cerr << "Error: Input arrays cannot be empty" << std::endl;
    return;
  }

  const uint32_t num_points = static_cast<uint32_t>(h_labels.size());
  const uint32_t num_columns = 3;

  if (h_points_flat.size() != num_points * num_columns) {
    std::cerr << "Error: Points array size mismatch. Expected " << num_points * num_columns << " got "
              << h_points_flat.size() << std::endl;
    return;
  }

  const uint32_t num_centroids = static_cast<uint32_t>(h_centroids_flat.size()) / num_columns;

  if (h_centroids_flat.size() % num_columns != 0) {
    std::cerr << "Error: Centroids array size must be multiple of 3" << std::endl;
    return;
  }

  if (num_centroids == 0) {
    std::cerr << "Error: No centroids provided" << std::endl;
    return;
  }

  std::cout << "GPU Clustering: points=" << num_points << ", centroids=" << num_centroids << ", columns=" << num_columns
            << std::endl;

  const uint32_t num_batches = (num_points + BATCH_SIZE - 1) / BATCH_SIZE;

  float* d_points = nullptr;
  float* d_centroids = nullptr;
  uint32_t* d_results = nullptr;
  uint32_t* d_labels_batch = nullptr;

  try {
    CUDA_CHECK(cudaMalloc(&d_points, h_points_flat.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, h_centroids_flat.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_results, BATCH_SIZE * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_labels_batch, BATCH_SIZE * sizeof(uint32_t)));
    CUDA_CHECK(
        cudaMemcpy(d_points, h_points_flat.data(), h_points_flat.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids_flat.data(), h_centroids_flat.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    Uniforms uniforms;
    uniforms.num_centroids = num_centroids;
    uniforms.num_columns = num_columns;

    for (uint32_t batch = 0; batch < num_batches; batch++) {
      uint32_t batch_start = batch * BATCH_SIZE;
      uint32_t current_batch_size = min(BATCH_SIZE, num_points - batch_start);

      uniforms.num_points = current_batch_size;

      uint32_t num_groups = (current_batch_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

      size_t shared_mem_size = num_columns * CHUNK_SIZE * sizeof(float);

      cluster_kernel_flat_column_major<<<num_groups, WORKGROUP_SIZE, shared_mem_size>>>(
          d_points + batch_start, d_centroids, d_results, uniforms);

      CUDA_CHECK(cudaGetLastError());

      CUDA_CHECK(cudaMemcpy(d_labels_batch, d_results, current_batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));

      std::copy(d_labels_batch, d_labels_batch + current_batch_size, h_labels.begin() + batch_start);

      std::cout << "  Batch " << batch + 1 << "/" << num_batches << " processed " << current_batch_size << " points"
                << std::endl;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

  } catch (const std::exception& e) {
    std::cerr << "Error during GPU clustering: " << e.what() << std::endl;

    if (d_points) cudaFree(d_points);
    if (d_centroids) cudaFree(d_centroids);
    if (d_results) cudaFree(d_results);
    if (d_labels_batch) cudaFree(d_labels_batch);

    throw;
  }

  if (d_points) cudaFree(d_points);
  if (d_centroids) cudaFree(d_centroids);
  if (d_results) cudaFree(d_results);
  if (d_labels_batch) cudaFree(d_labels_batch);
}

}  // namespace splat
