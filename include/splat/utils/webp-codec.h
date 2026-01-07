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

#include <cstdint>
#include <vector>

namespace splat::webpcodec {

/**
 * @brief Decodes a WebP image to RGBA format
 *
 * This function decodes a WebP-encoded image into raw RGBA pixel data.
 * It uses the WebP library's RGBA decoder to convert the compressed WebP data
 * into an uncompressed RGBA byte array.
 *
 * @param webp Input WebP-encoded image data as a vector of bytes
 *
 * @return std::tuple containing:
 *         - std::vector<uint8_t>: Decoded RGBA pixel data, arranged as
 *           [R0, G0, B0, A0, R1, G1, B1, A1, ...] for each pixel row-major
 *         - int: Width of the decoded image in pixels
 *         - int: Height of the decoded image in pixels
 *
 */
std::tuple<std::vector<uint8_t>, int, int> decodeRGBA(const std::vector<uint8_t>& webp);

/**
 * @brief Encodes RGBA image data to lossless WebP format
 *
 * This function compresses raw RGBA pixel data into a lossless WebP image.
 * It uses WebP's lossless encoding mode which preserves exact pixel values
 * while providing good compression for RGBA images.
 *
 * @param rgba Input RGBA pixel data as a vector of bytes, arranged as
 *             [R0, G0, B0, A0, R1, G1, B1, A1, ...] for each pixel row-major
 * @param width Width of the input image in pixels
 * @param height Height of the input image in pixels
 * @param stride Number of bytes between consecutive rows. If 0, stride is
 *               calculated as width * 4 (standard RGBA layout)
 *
 * @return std::vector<uint8_t> Lossless WebP-encoded image data
 *
 */
std::vector<uint8_t> encodeLosslessRGBA(const std::vector<uint8_t>& rgba, int width, int height, int stride = 0);

}  // namespace splat::webpcodec
