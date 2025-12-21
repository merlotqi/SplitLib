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

#include <splat/splat_version.h>

#include <splat/types.h>
#include <splat/logger.h>
#include <splat/readers/ply_reader.h>
#include <splat/writers/sog_writer.h>

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>

#include <absl/strings/str_split.h>
#include <absl/strings/numbers.h>
#include <absl/strings/strip.h>
#include <absl/strings/match.h>

#include "process.h"
#include "gpudevice.h"


namespace fs = std::filesystem;

using namespace splat;

struct File {
  std::string filename;
  std::vector<ProcessAction> processActions;
};

ABSL_FLAG(bool, overwrite, false, "Overwrite output file if it exists");
ABSL_FLAG(bool, quiet, false, "Suppress non-error output");
ABSL_FLAG(std::string, iterations, "10", "Iterations for SOG SH compression (more=better). Default: 10");
ABSL_FLAG(bool, list_gpus, false, "List available GPU adapters");
ABSL_FLAG(std::string, gpu, "-1", "Select device: index or 'cpu'");
ABSL_FLAG(std::string, lod_select, "", "Comma-separated LOD levels");
ABSL_FLAG(std::string, viewer_settings, "", "HTML viewer settings JSON file");
ABSL_FLAG(bool, unbundled, false, "Generate unbundled HTML viewer");
ABSL_FLAG(std::string, lod_chunk_count, "512", "Gaussians per LOD chunk in K");
ABSL_FLAG(std::string, lod_chunk_extent, "16", "LOD chunk size in world units");

static std::string usage = R"(
Transform and Filter Gaussian Splats
====================================

USAGE
  splat-transform [GLOBAL] input [ACTIONS]  ...  output [ACTIONS]

  • Input files become the working set; ACTIONS are applied in order.
  • The last file is the output; actions after it modify the final result.

SUPPORTED INPUTS
    .ply   .compressed.ply   .sog   meta.json   .ksplat   .splat   .spz   .mjs   .lcc

SUPPORTED OUTPUTS
    .ply   .compressed.ply   .sog   meta.json   .csv   .html

ACTIONS (can be repeated, in any order)
    -t, --translate        <x,y,z>          Translate splats by (x, y, z)
    -r, --rotate           <x,y,z>          Rotate splats by Euler angles (x, y, z), in degrees
    -s, --scale            <factor>         Uniformly scale splats by factor
    -H, --filter-harmonics <0|1|2|3>        Remove spherical harmonic bands > n
    -N, --filter-nan                        Remove Gaussians with NaN or Inf values
    -B, --filter-box       <x,y,z,X,Y,Z>    Remove Gaussians outside box (min, max corners)
    -S, --filter-sphere    <x,y,z,radius>   Remove Gaussians outside sphere (center, radius)
    -V, --filter-value     <name,cmp,value> Keep splats where <name> <cmp> <value>
                                              cmp ∈ {lt,lte,gt,gte,eq,neq}
    -p, --params           <key=val,...>    Pass parameters to .mjs generator script
    -l, --lod              <n>              Specify the level of detail, n >= 0.

GLOBAL OPTIONS
    -h, --help                              Show this help and exit
    -v, --version                           Show version and exit
    -q, --quiet                             Suppress non-error output
    -w, --overwrite                         Overwrite output file if it exists
    -i, --iterations       <n>              Iterations for SOG SH compression (more=better). Default: 10
    -L, --list-gpus                         List available GPU adapters and exit
    -g, --gpu              <n|cpu>          Select device for SOG compression: GPU adapter index | 'cpu'
    -E, --viewer-settings  <settings.json>  HTML viewer settings JSON file
    -U, --unbundled                         Generate unbundled HTML viewer with separate files
    -O, --lod-select       <n,n,...>        Comma-separated LOD levels to read from LCC input
    -C, --lod-chunk-count  <n>              Approximate number of Gaussians per LOD chunk in K. Default: 512
    -X, --lod-chunk-extent <n>              Approximate size of an LOD chunk in world units (m). Default: 16

EXAMPLES
    # Scale then translate
    splat-transform bunny.ply -s 0.5 -t 0,0,10 bunny-scaled.ply

    # Merge two files with transforms and compress to SOG format
    splat-transform -w cloudA.ply -r 0,90,0 cloudB.ply -s 2 merged.sog

    # Generate unbundled HTML viewer with separate CSS, JS and SOG files
    splat-transform -U bunny.ply bunny-viewer.html

    # Generate synthetic splats using a generator script
    splat-transform gen-grid.mjs -p width=500,height=500,scale=0.1 grid.ply

    # Generate LOD with custom chunk size and node split size
    splat-transform -O 0,1,2 -C 1024 -X 32 input.lcc output/lod-meta.json
)";


static std::tuple<std::vector<File>, Options> parseArguments(int argc, char** argv) {
  auto parseNumber = [](absl::string_view value) {
    float result;
    if (!absl::SimpleAtof(value, &result)) {
      throw std::runtime_error("Invalid number value" + std::string(value));
    }
    return result;
  };

  auto parseInteger = [](absl::string_view value) {
    int result;
    if (!absl::SimpleAtoi(value, &result)) {
      throw std::runtime_error("Invalid number value" + std::string(value));
    }
    return result;
  };

  auto parseVec3 = [parseNumber](absl::string_view value) {
    std::vector<absl::string_view> parts = absl::StrSplit(value, ',');
    if (parts.size() != 3) {
      throw std::runtime_error("Invalid Vec3 value: " + std::string(value));
    }
    return Eigen::Vector3f(parseNumber(parts[0]), parseNumber(parts[1]), parseNumber(parts[2]));
  };

  auto parseComparator = [](absl::string_view value) -> std::string {
    static const std::set<absl::string_view> valid = {"lt", "lte", "gt", "gte", "eq", "neq"};
    if (valid.find(value) == valid.end()) {
      throw std::runtime_error("Invalid comparator value: " + std::string(value));
    }
    return std::string(value);
  };

  absl::SetProgramUsageMessage(
      "Transform and Filter Gaussian Splats\nUSAGE: splat-transform [GLOBAL] input [ACTIONS] ... output [ACTIONS]");

  std::vector<char*> remaining_args = absl::ParseCommandLine(argc, argv);

  Options options;
  options.overwrite = absl::GetFlag(FLAGS_overwrite);
  options.quiet = absl::GetFlag(FLAGS_quiet);
  options.listGpus = absl::GetFlag(FLAGS_list_gpus);
  options.unbundled = absl::GetFlag(FLAGS_unbundled);
  options.viewerSettingsPath = absl::GetFlag(FLAGS_viewer_settings);
  options.iterations = parseInteger(absl::GetFlag(FLAGS_iterations));
  options.lodChunkCount = parseInteger(absl::GetFlag(FLAGS_lod_chunk_count));
  options.lodChunkExtent = parseInteger(absl::GetFlag(FLAGS_lod_chunk_extent));

  // Parse gpu option - can be a number or "cpu"
  std::string gpu_val = absl::GetFlag(FLAGS_gpu);
  if (gpu_val == "cpu") {
    options.device = -2;
  } else {
    options.device = parseInteger(gpu_val);
  }

  std::string lod_s = absl::GetFlag(FLAGS_lod_select);
  if (!lod_s.empty()) {
    for (auto s : absl::StrSplit(lod_s, ',', absl::SkipEmpty())) {
      options.lodSelect.push_back(parseInteger(s));
    }
  }


  std::vector<File> files;
  for (size_t i = 1; i < remaining_args.size(); ++i) {
    absl::string_view arg = remaining_args[i];

    if (!absl::StartsWith(arg, "-")) {
      files.push_back({std::string(arg), {}});
    } else if (!files.empty()) {
      File& current = files.back();

      absl::string_view name = arg;
      while (absl::ConsumePrefix(&name, "-")) {
      }

      auto getNextValue = [&]() {
        if (i + 1 >= remaining_args.size())
          throw std::runtime_error("Action " + std::string(arg) + " requires a value.");
        return absl::string_view(remaining_args[++i]);
      };

      if (name == "t" || name == "translate") {
        current.processActions.push_back(Translate{parseVec3(getNextValue())});
      } else if (name == "r" || name == "rotate") {
        current.processActions.push_back(Rotate{parseVec3(getNextValue())});
      } else if (name == "s" || name == "scale") {
        current.processActions.push_back(Scale{parseNumber(getNextValue())});
      } else if (name == "N" || name == "filter-nan") {
        current.processActions.push_back(FilterNaN{});
      } else if (name == "V" || name == "filter-value") {
        std::vector<std::string> parts = absl::StrSplit(getNextValue(), ',');
        if (parts.size() != 3) throw std::runtime_error("Invalid filter-value");
        current.processActions.push_back(FilterByValue{parts[0], parseComparator(parts[1]), parseNumber(parts[2])});
      } else if (name == "H" || name == "filter-harmonics") {
        current.processActions.push_back(FilterBands{parseInteger(getNextValue())});
      } else if (name == "B" || name == "filter-box") {
        std::vector<absl::string_view> parts = absl::StrSplit(getNextValue(), ',');
        if (parts.size() != 6) throw std::runtime_error("Invalid filter-box");

        float defaults[] = {-INFINITY, -INFINITY, -INFINITY, INFINITY, INFINITY, INFINITY};
        float values[6];
        for (int j = 0; j < 6; ++j) {
          if (parts[j].empty() || parts[j] == "-")
            values[j] = defaults[j];
          else
            values[j] = parseNumber(parts[j]);
        }
        current.processActions.push_back(
            FilterBox{{values[0], values[1], values[2]}, {values[3], values[4], values[5]}});
      } else if (name == "S" || name == "filter-sphere") {
        std::vector<absl::string_view> parts = absl::StrSplit(getNextValue(), ',');
        if (parts.size() != 4) throw std::runtime_error("Invalid filter-sphere");
        current.processActions.push_back(
            FilterSphere{{parseNumber(parts[0]), parseNumber(parts[1]), parseNumber(parts[2])}, parseNumber(parts[3])});
      } else if (name == "p" || name == "params") {
        for (auto p : absl::StrSplit(getNextValue(), ',')) {
          std::pair<std::string, std::string> kv = absl::StrSplit(p, absl::MaxSplits('=', 1));
          current.processActions.push_back(Param{kv.first, kv.second});
        }
      } else if (name == "l" || name == "lod") {
        current.processActions.push_back(Lod{parseInteger(getNextValue())});
      }
    }
  }

  return {files, options};
}

static std::string getOutputFormat(std::string filename) {
  if (absl::EndsWithIgnoreCase(filename, ".csv")) {
    return "csv";
  }
  if (absl::EndsWithIgnoreCase(filename, "lod-meta.json")) {
    return "lod";
  }
  if (absl::EndsWithIgnoreCase(filename, ".sog") || absl::EndsWithIgnoreCase(filename, "meta.json")) {
    return "sog";
  }
  if (absl::EndsWithIgnoreCase(filename, ".compressed.ply")) {
    return "compressed-ply";
  }
  if (absl::EndsWithIgnoreCase(filename, ".ply")) {
    return "ply";
  }
  if (absl::EndsWithIgnoreCase(filename, ".html")) {
    return "html";
  }

  throw std::runtime_error("Unsupported output file type: " + std::string(filename));
}

static std::vector<DataTable> readFile(const std::string& filename, const Options& options,
                                       const std::vector<Param>& params) {

}

static void writeFile(const std::string& filename, const DataTable& dataTable, DataTable* envDataTable, const Options& options)
{

}

static bool isGSDataTable(const DataTable& dataTable) {
  static std::vector<std::string> required_columns = {
      "x",      "y",      "z",      "rot_0', 'rot_1', 'rot_2', 'rot_3", "scale_0", "scale_1", "scale_2", "f_dc_0",
      "f_dc_1", "f_dc_2", "opacity"};

  bool gs = std::all_of(required_columns.begin(), required_columns.end(),
                        [&](const std::string& c) { return dataTable.hasColumn(c); });
  return gs;
}

static DataTable combine(const std::vector<DataTable>& dataTables) { return DataTable(); }


int main(int argc, char** argv) {
  std::chrono::time_point startTime = std::chrono::high_resolution_clock::now();

  auto [files, options] = parseArguments(argc, argv);

  Logger::instance().setQuiet(options.quiet);
  LOG_INFO("splat-transform v%s", std::string(SPLAT_VERSION));

  // show version and exit
  if (options.version) {
    exit(0);
  }

  if (options.listGpus) {
    LOG_INFO("Enumerating available GPU adapters...\n");
    try {
      std::vector<AdapterInfo> adapters = enumerateAdapters();

      if (adapters.empty()) {
        LOG_INFO("No GPU adapters found.");
        LOG_INFO("This could mean:");
        LOG_INFO("  - Graphics drivers need to be updated");
        LOG_INFO("  - Your system does not support the required graphics API");
      } else {
        for (const auto& adapter : adapters) {
          LOG_INFO("[%d] %s", adapter.index, adapter.name);
        }
        LOG_INFO("\nUse -g <index> to select a specific GPU adapter.");
      }
    } catch (const std::exception& err) {
      LOG_ERROR("Failed to enumerate GPU adapters: %s", err.what());
    }
    std::exit(0);
  }

  // invalid args or show help
  if (files.size() < 2 || options.help) {
    LOG_ERROR(usage);
    std::exit(1);
  }

  std::vector<File> inputArgs(files.begin(), files.end() - 1);
  File outputArg = files.back();

  fs::path outputFilename = fs::absolute(outputArg.filename);
  std::string outputFormat = getOutputFormat(outputFilename.string());

  if (options.overwrite) {
    std::error_code ec;
    fs::create_directories(outputFilename.parent_path(), ec);
    if (ec) {
      LOG_ERROR("Failed to create directory: {}", ec.message());
      std::exit(1);
    }
  } else {
    if (fs::exists(outputFilename)) {
      LOG_ERROR("File '{}' already exists. Use -w option to overwrite.", outputFilename.string());
      std::exit(1);
    }

    if (outputFormat == "html" && options.unbundled) {
      fs::path outputDir = outputFilename.parent_path();
      std::string baseFilename = outputFilename.stem().string();

      std::vector<fs::path> filesToCheck = {outputDir / "index.css", outputDir / "index.js",
                                            outputDir / (baseFilename + ".sog")};

      for (const auto& file : filesToCheck) {
        if (fs::exists(file)) {
          LOG_ERROR("File '{}' already exists. Use -w option to overwrite.", file.string());
          std::exit(1);
        }
      }
    }
  }

  try {

      std::vector<DataTable> inputDataTables;

    for (const auto& inputArg : inputArgs) {
      std::vector<Param> params;

      std::vector<DataTable> dts = readFile(inputArg.filename, options, params);

      for (auto& dt : dts) {
        if (dt.getNumRows() == 0 || !isGSDataTable(dt)) {
          throw std::runtime_error("Unsupported data in file: " + inputArg.filename);
        }

        dt = processDataTable(dt, inputArg.processActions);
        inputDataTables.push_back(dt);
      }
    }

    std::vector<DataTable> envDataTables;
    std::vector<DataTable> nonEnvDataTables;

    // special-case the environment dataTable
    for (const auto& dt : inputDataTables) {
      if (dt.hasColumn("lod") && dt.getColumnByName("lod").every<int>(-1)) envDataTables.push_back(dt);
      if (!dt.hasColumn("lod") || (dt.hasColumn("lod") && dt.getColumnByName("lod").some<int>(-1)))
        nonEnvDataTables.push_back(dt);
    }

    // combine inputs into a single output dataTable
    DataTable dataTable;
    if (!nonEnvDataTables.empty()) {
      dataTable = processDataTable(combine(nonEnvDataTables), outputArg.processActions);
    }

    if (dataTable.getNumRows() == 0) {
      throw std::runtime_error("No splats to write");
    }

    DataTable envDataTable;
    if (!envDataTables.empty()) {
      envDataTable = processDataTable(combine(envDataTables), outputArg.processActions);
    }

    LOG_INFO("Loaded %d gaussians", dataTable.getNumRows());

    writeFile(outputFilename.string(), dataTable, &envDataTable, options);


  } catch (const std::exception&e) {
    LOG_ERROR(e.what());
    std::exit(1);
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime - startTime;
  LOG_INFO("done in {:.6f}s", elapsed.count());

  std::exit(0);
}
