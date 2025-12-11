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

namespace splat {

#define CHUNK_SIZE (128u)
#define THREADS_PER_BLOCK (64u)

__device__ float calcDistanceSqr(const float* point, const float* sharedChunk, uint32_t centroidIndexInChunk,
                                 uint32_t numColumns) {
  float result = 0.0f;
  uint32_t ci = centroidIndexInChunk * numColumns;

  for (uint32_t i = 0; i < numColumns; i++) {
    float v = point[i] - sharedChunk[ci + i];
    result += v * v;
  }
  return result;
}

__global__ void clusterKernel(const float* points, const float* centroids, uint32_t* results, uint32_t numPoints,
                              uint32_t numCentroids, uint32_t numColumns) {
  extern __shared__ float sharedChunk[];

  uint32_t pointIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (pointIndex >= numPoints) {
    return;
  }

  const uint32_t MAX_COLS = 32u;
  float pointReg[MAX_COLS];

  if (numColumns > MAX_COLS) {
    return;
  }

  for (uint32_t i = 0; i < numColumns; i++) {
    pointReg[i] = points[pointIndex * numColumns + i];
  }

  float mind = FLT_MAX;
  uint32_t mini = 0;

  uint32_t numChunks = (numCentroids + CHUNK_SIZE - 1u) / CHUNK_SIZE;
  uint32_t numThreads = blockDim.x;

  for (uint32_t i = 0; i < numChunks; i++) {
    uint32_t centroidStartGlobal = i * CHUNK_SIZE;
    uint32_t chunkSize = min(CHUNK_SIZE, numCentroids - centroidStartGlobal);

    uint32_t centroidRowIndex = centroidStartGlobal + threadIdx.x;

    if (centroidRowIndex < numCentroids) {
      uint32_t src = centroidRowIndex * numColumns;
      uint32_t dst = threadIdx.x * numColumns;

      for (uint32_t c = 0u; c < numColumns; c++) {
        if (dst + c < CHUNK_SIZE * numColumns) {
          sharedChunk[dst + c] = centroids[src + c];
        }
      }
    }

    __syncthreads();

    if (pointIndex < numPoints) {
      for (uint32_t c = 0u; c < chunkSize; c++) {
        float d = calcDistanceSqr(pointReg, sharedChunk, c, numColumns);

        if (d < mind) {
          mind = d;
          mini = centroidStartGlobal + c;
        }
      }
    }

    if (i < numChunks - 1u) {
      __syncthreads();
    }
  }
  if (pointIndex < numPoints) {
    results[pointIndex] = mini;
  }
}

std::vector<uint32_t> gpu_cluster(const std::vector<float>& h_points, const std::vector<float>& h_centroids,
                                  uint32_t numPoints, uint32_t numCentroids, uint32_t numColumns) {
  if (numPoints == 0 || numCentroids == 0 || numColumns == 0) {
    return {};
  }
  if (h_points.size() != numPoints * numColumns || h_centroids.size() != numCentroids * numColumns) {
    throw std::runtime_error("Input vector size does not match N*D or K*D.");
  }

  std::vector<uint32_t> h_results(numPoints);
  float *d_points, *d_centroids;
  uint32_t* d_results;

  size_t pointsSize = h_points.size() * sizeof(float);
  size_t centroidsSize = h_centroids.size() * sizeof(float);
  size_t resultsSize = h_results.size() * sizeof(uint32_t);

  if (cudaMalloc(&d_points, pointsSize) != cudaSuccess || cudaMalloc(&d_centroids, centroidsSize) != cudaSuccess ||
      cudaMalloc(&d_results, resultsSize) != cudaSuccess) {
    throw std::runtime_error("CUDA memory allocation failed.");
  }

  if (cudaMemcpy(d_points, h_points.data(), pointsSize, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_centroids, h_centroids.data(), centroidsSize, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_results); 
    throw std::runtime_error("CUDA HtoD copy failed.");
  }

  const int threadsPerBlock = THREADS_PER_BLOCK;  // 64
  const int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

  const size_t sharedMemSize = CHUNK_SIZE * numColumns * sizeof(float);

  clusterKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_points, d_centroids, d_results, numPoints,
                                                                   numCentroids, numColumns);

  cudaError_t lastError = cudaGetLastError();
  if (lastError != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(lastError) << std::endl;
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_results);
    throw std::runtime_error("CUDA Kernel launch failed.");
  }

  if (cudaMemcpy(h_results.data(), d_results, resultsSize, cudaMemcpyDeviceToHost) != cudaSuccess) {
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_results);
    throw std::runtime_error("CUDA DtoH copy failed.");
  }

  cudaFree(d_points);
  cudaFree(d_centroids);
  cudaFree(d_results);

  return h_results;
}

}  // namespace splat
