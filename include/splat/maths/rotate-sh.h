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

#include <Eigen/Dense>

namespace splat {

/**
 * @struct RotateSH
 * @brief Rotation matrices for spherical harmonics up to band 3
 *
 * This structure stores precomputed rotation matrices for spherical harmonics (SH)
 * used to efficiently rotate SH coefficients. It supports rotation of SH coefficients
 * up to band 3 (including bands 0-3), which corresponds to 16 coefficients per color channel.
 *
 * The rotation matrices are constructed from a 3x3 rotation matrix and can be applied
 * to transform spherical harmonic coefficients under 3D rotations.
 */
struct RotateSH {
  float sh1[3][3];  ///< Rotation matrix for band 1 SH coefficients (3x3)
  float sh2[5][5];  ///< Rotation matrix for band 2 SH coefficients (5x5)
  float sh3[7][7];  ///< Rotation matrix for band 3 SH coefficients (7x7)

  /**
   * @brief Apply rotation to spherical harmonic coefficients
   *
   * Rotates the input spherical harmonic coefficients using the precomputed
   * rotation matrices. The function can operate in-place (if src is empty)
   * or copy from source to result.
   *
   * @param result Output vector for rotated SH coefficients. Must have size
   *               equal to the number of coefficients being rotated (typically
   *               16 per color channel for bands 0-3).
   * @param src Source SH coefficients to rotate. If empty, the function
   *            uses result as both source and destination (in-place rotation).
   *            If provided, src must have the same size as result.
   *
   * @note The function assumes coefficients are stored in standard SH ordering:
   *       band 0 (1), band 1 (3), band 2 (5), band 3 (7) = total 16 coefficients.
   * @note Band 0 (constant term) is not rotated as it's invariant under rotation.
   */
  void apply(std::vector<float>& result, std::vector<float> src = {});

  /**
   * @brief Construct rotation matrices from a 3x3 rotation matrix
   *
   * Initializes the SH rotation matrices (bands 1-3) based on the provided
   * 3x3 rotation matrix. The rotation matrix should be orthogonal (rotation only,
   * no scaling or shearing).
   *
   * @param mat 3x3 rotation matrix (typically from Eigen::Quaternionf or Eigen::Matrix3f)
   *
   * @note The constructor computes Wigner D-matrices for each band using
   *       the provided rotation matrix.
   * @note Band 0 rotation matrix is identity and not stored (band 0 is rotationally invariant).
   */
  RotateSH(const Eigen::Matrix3f& mat);
};

}  // namespace splat
