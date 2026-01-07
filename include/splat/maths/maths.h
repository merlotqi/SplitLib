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

#include <cmath>

namespace splat {

/**
 * @brief Sigmoid activation function
 *
 * Computes the sigmoid (logistic) function: σ(x) = 1 / (1 + exp(-x))
 * This function maps real values to the range (0, 1) and is commonly used
 * in machine learning for probabilities and activations.
 *
 * @tparam T Numeric type (float, double, etc.)
 * @param x Input value
 * @return Sigmoid-transformed value in range (0, 1)
 *
 * @note For Gaussian splatting, this is typically used to convert raw opacity
 *       values to valid opacity in [0, 1] range.
 * @note Uses double-precision exponential for accuracy even with float input.
 */
template <typename T>
inline T sigmoid(T x) {
  return 1.0 / (1.0 + exp(-x));
}

/**
 * @brief Simple deterministic pseudo-random number generator
 *
 * Generates a pseudo-random float in the range [0, 1) using a linear congruential
 * generator (LCG). The generator is deterministic and thread-unsafe due to static
 * internal state.
 *
 * @return Random float in range [0, 1)
 *
 * @note This is a simple PRNG suitable for debugging, testing, or situations
 *       where reproducibility is more important than statistical quality.
 * @note Not suitable for cryptographic or high-quality statistical applications.
 * @note The generator uses the constants from POSIX rand() implementation:
 *       multiplier = 1103515245, increment = 12345, modulus = 2^31.
 * @note Thread-unsafe: uses static internal state.
 */
inline float simple_random() {
  static unsigned int seed = 42;  ///< Initial seed (can be changed for different sequences)
  seed = (1103515245ULL * seed + 12345) & 0x7FFFFFFF;
  return (float)seed / (float)0x7FFFFFFF;
}

}  // namespace splat
