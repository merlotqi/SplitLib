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
#include <string_view>
#include <iostream>
#include <mutex>
#include <filesystem>
#include <absl/strings/str_format.h>

namespace splat {

enum class logLevel {
  silent,
  normal
};

class Logger {
  logLevel level = logLevel::normal;
  std::mutex log_mutex;

  Logger() = default;

  template <typename... Args>
  void log_internal(std::string_view prefix, std::string_view file, int line, std::string_view format, Args&&... args) {
    if (level == logLevel::silent) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    std::string formatted_msg;
    if constexpr (sizeof...(args) == 0) {
      formatted_msg = std::string(format);
    } else {
      formatted_msg = absl::StrFormat(std::string(format), std::forward<Args>(args)...);
    }

    std::filesystem::path file_path(file);
    std::cout << "[" << prefix << "] " << file_path.filename().string() << ":" << line << " > " << formatted_msg
              << std::endl;
  }

 public:
  static Logger& instance()
  {
    static Logger instance;
    return instance;
  }

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  void setQuiet(bool quiet) { level = quiet ? logLevel::silent : logLevel::normal; }

  template <typename... Args>
  void info(const char* file, int line, std::string_view format, Args&&... args) {
    log_internal("INFO", file, line, format, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void warn(const char* file, int line, std::string_view format, Args&&... args) {
    log_internal("WARN", file, line, format, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void error(const char* file, int line, std::string_view format, Args&&... args) {
    log_internal("ERROR", file, line, format, std::forward<Args>(args)...);
  }
};

}  // namespace splat

#define LOG_INFO(format, ...) splat::Logger::instance().info(__FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) splat::Logger::instance().warn(__FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) splat::Logger::instance().error(__FILE__, __LINE__, format, ##__VA_ARGS__)
