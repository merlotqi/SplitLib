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

#include <string>

namespace splat {

namespace logger {

enum Level { INFO = 0, WARN = 1, ERROR = 2, DEBUG = 3 };

void setLevel(Level level);
Level getLevel();

#ifdef _WIN32
void addOutputFile(const std::wstring& path);
#else
void addOutputFile(const std::string& path);
#endif

void closeOutputFile();

void info(const char* fmt, ...);
void warn(const char* fmt, ...);
void error(const char* fmt, ...);
void debug(const char* fmt, ...);

}  // namespace logger
}  // namespace splat

#define INFO(fmt, ...) splat::logger::info(fmt, ##__VA_ARGS__)
#define WARN(fmt, ...) splat::logger::warn(fmt, ##__VA_ARGS__)
#define ERROR(fmt, ...) splat::logger::error(fmt, ##__VA_ARGS__)
#define DEBUG(fmt, ...) splat::logger::debug(fmt, ##__VA_ARGS__)
