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

#include <splat/logger.h>
#include <stdarg.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;

namespace splat {
namespace logger {

static std::shared_ptr<std::ofstream> fout = nullptr;
static std::mutex mtx;
static Level current_level = INFO;

static std::string getCurrentTime() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
  ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
  return ss.str();
}

static std::string getThreadId() {
  std::stringstream ss;
  ss << std::this_thread::get_id();
  return ss.str();
}

static std::string format_string(const char* fmt, va_list args) {
  va_list args_copy;
  va_copy(args_copy, args);

  int size = vsnprintf(nullptr, 0, fmt, args_copy);
  va_end(args_copy);

  if (size < 0) {
    return "";
  }

  std::string result(size, '\0');
  vsnprintf(result.data(), size + 1, fmt, args);

  return result;
}

void setLevel(Level level) {
  std::lock_guard<std::mutex> lock(mtx);
  current_level = level;
}

Level getLevel() {
  std::lock_guard<std::mutex> lock(mtx);
  return current_level;
}

#ifdef _WIN32
void addOutputFile(const std::wstring& path) {
  std::lock_guard<std::mutex> lock(mtx);

  fs::path dir = fs::path(path).parent_path();
  if (!dir.empty() && !fs::exists(dir)) {
    fs::create_directories(dir);
  }

  if (fout != nullptr && fout->is_open()) {
    fout->close();
  }

  fout.reset(new std::ofstream());
  fout->open(path, std::ios::app);

  if (!fout->is_open()) {
    std::cerr << "Failed to open log file: " << std::string(path.begin(), path.end()) << std::endl;
    fout = nullptr;
  }
}
#else
void addOutputFile(const std::string& path) {
  std::lock_guard<std::mutex> lock(mtx);

  fs::path dir = fs::path(path).parent_path();
  if (!dir.empty() && !fs::exists(dir)) {
    fs::create_directories(dir);
  }

  if (fout != nullptr && fout->is_open()) {
    fout->close();
  }

  fout.reset(new std::ofstream());
  fout->open(path, std::ios::app);

  if (!fout->is_open()) {
    std::cerr << "Failed to open log file: " << path << std::endl;
    fout = nullptr;
  }
}
#endif

void closeOutputFile() {
  std::lock_guard<std::mutex> lock(mtx);
  if (fout != nullptr && fout->is_open()) {
    fout->close();
  }
  fout = nullptr;
}

static void write_log(const char* level, const char* fmt, va_list args) {
  std::lock_guard<std::mutex> lock(mtx);

  std::string timestamp = getCurrentTime();
  std::string thread_id = getThreadId();
  std::string message = format_string(fmt, args);

  std::stringstream log_line;
  log_line << timestamp << " [" << level << "] "
           << "[Thread:" << thread_id << "] " << message << std::endl;

  std::string log_str = log_line.str();

  if (fout != nullptr && fout->is_open()) {
    *fout << log_str;
    fout->flush();
  }

  if (strcmp(level, "WARN") == 0 || strcmp(level, "ERROR") == 0) {
    std::cerr << log_str;
  }

  if (strcmp(level, "INFO") == 0) {
    std::cout << log_str;
  }
}

void info(const char* fmt, ...) {
  if (current_level > INFO) return;

  va_list args;
  va_start(args, fmt);
  write_log("INFO", fmt, args);
  va_end(args);
}

void warn(const char* fmt, ...) {
  if (current_level > WARN) return;

  va_list args;
  va_start(args, fmt);
  write_log("WARN", fmt, args);
  va_end(args);
}

void error(const char* fmt, ...) {
  if (current_level > ERROR) return;

  va_list args;
  va_start(args, fmt);
  write_log("ERROR", fmt, args);
  va_end(args);
}

void debug(const char* fmt, ...) {
#ifdef _DEBUG
  va_list args;
  va_start(args, fmt);
  write_log("DEBUG", fmt, args);
  va_end(args);
#else
  (void)fmt;
#endif
}

}  // namespace logger
}  // namespace splat
