/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
https://github.com/FrancoisCarouge/Kalman

SPDX-License-Identifier: Unlicense

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org> */

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <print>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace {
bool is_scientific_number(const std::string &s) {
  return s.find("e-") != std::string::npos;
}

std::string to_natural(const std::string &scientific_number) {
  double value{std::stod(scientific_number)};
  std::ostringstream oss;
  oss << std::fixed
      << std::setprecision(std::numeric_limits<double>::max_digits10) << value;

  std::string natural_number{oss.str()};
  std::size_t decimalPos{natural_number.find('.')};
  if (decimalPos != std::string::npos) {
    natural_number.erase(natural_number.find_last_not_of('0') + 1,
                         std::string::npos);
    if (natural_number.back() == '.') {
      natural_number.pop_back();
    }
  }

  return natural_number;
}

void convert_scientific_to_natural(std::vector<std::string> &lines) {
  for (auto &line : lines) {
    std::stringstream ss(line);
    std::string field;
    std::string result;
    bool first{true};

    while (std::getline(ss, field, ',')) {
      if (!first) {
        result += ',';
      }
      first = false;

      if (is_scientific_number(field)) {
        result += ' ';
        result += to_natural(field);
      } else {
        result += field;
      }
    }

    line = result;
  }
}

// Key of the line based on the first two comma-separated value fields.
std::string_view key_of(std::string_view line, std::size_t field_count) {
  std::size_t position{std::string_view::npos};

  for (std::size_t i{0}; i < field_count; ++i) {
    std::size_t next_field{line.find(", ", ++position)};

    if (next_field == std::string_view::npos) {
      return line.substr(0, position); // No enough fields.
    }

    position = next_field;
  }

  return line.substr(0, position);
}

void write(const std::filesystem::path &file_path,
           const std::vector<std::string> &lines) {
  // Atomically write the file in place.
  std::filesystem::path temporary{file_path.string() + ".tmp"};
  {
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) {
      std::println("Failed to open temporary file: '{}'.", temporary.string());
      std::exit(EXIT_FAILURE);
    }

    for (auto &&line : lines) {
      std::println(output, "{}", line);
    }
  }

  std::error_code status;
  std::filesystem::rename(temporary, file_path, status);
  if (status) {
    std::println("Failed to replace file: '{}'.", status.message());
    std::filesystem::remove(temporary);
    std::exit(EXIT_FAILURE);
  }
}

std::vector<std::string> read(const std::filesystem::path &file_path) {
  std::println("Processing: '{}'...", file_path.string());

  std::ifstream input(file_path);
  if (!input) {
    std::println("Failed to open file: '{}'.", file_path.string());
    std::exit(EXIT_FAILURE);
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(input, line)) {
    lines.push_back(line);
  }
  return lines;
}
} // namespace

int main(int argument_count, char **argument_value) {
  if (argument_count != 2) {
    std::println("Usage: {} <result path>", argument_value[0]);
    return EXIT_FAILURE;
  }

  std::filesystem::path bin_directory{std::filesystem::current_path()};
  std::filesystem::path result_directory{argument_value[1]};
  std::println("Current working directory: '{}'.",
               std::filesystem::current_path().string());
  std::println("Binary directory: '{}'.", bin_directory.string());
  std::println("Result directory: '{}'.", result_directory.string());

  for (auto &&entry : std::filesystem::directory_iterator(bin_directory)) {
    std::string filename{entry.path().filename().string()};

    if (entry.is_regular_file() &&
        (filename.starts_with("kalman_benchmark_"))) {
      std::string command{".\\\\" + filename};
      std::cout.flush();
      std::println("Running: '{}'...", command);

      if (int status{std::system(command.c_str())}; status != 0) {
        std::println("Program: '{}' failed with code: '{}'.", command, status);
        return EXIT_FAILURE;
      }
    }
  }

  for (std::string filename : {"time.csv"}) {
    std::filesystem::path file_path{result_directory / filename};
    std::vector<std::string> lines{read(file_path)};
    convert_scientific_to_natural(lines);
    std::ranges::sort(lines);

    // Deduplicate the results.
    const auto [first, last]{std::ranges::unique(
        lines, [](const std::string &lhs, const std::string &rhs) {
          return key_of(lhs, 5) == key_of(rhs, 5);
        })};
    lines.erase(first, last);

    write(file_path, lines);
  }

  for (std::string filename : {"size.csv"}) {
    std::filesystem::path file_path{result_directory / filename};
    std::vector<std::string> lines{read(file_path)};
    std::ranges::sort(lines);

    // Deduplicate the results.
    const auto [first, last]{std::ranges::unique(
        lines, [](const std::string &lhs, const std::string &rhs) {
          return key_of(lhs, 4) == key_of(rhs, 4);
        })};
    lines.erase(first, last);

    write(file_path, lines);
  }

  std::filesystem::current_path(result_directory);
  std::println("Current working directory: '{}'.",
               std::filesystem::current_path().string());

  // Plot.
  for (auto &&entry : std::filesystem::directory_iterator(bin_directory)) {
    std::string filepath{entry.path().string()};
    if (entry.is_regular_file() && (filepath.ends_with(".plt"))) {
      std::string command{
          "start \"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\" " + filepath};
      std::println("Plotting: '{}'...", command);

      if (int status{std::system(command.c_str())}; status != 0) {
        std::println("Program: '{}' failed with code: '{}'.", command, status);
        return EXIT_FAILURE;
      }
    }
  }

  return EXIT_SUCCESS;
}
