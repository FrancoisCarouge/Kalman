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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>
#include <string>
#include <system_error>
#include <vector>

namespace {
// Key of the line based on the first two comma-separated value fields.
std::string_view key_of(std::string_view line) {
  if (std::size_t position{line.find(", ")};
      position != std::string_view::npos) {
    return line.substr(0, line.find(", ", ++position));
  }

  return line;
}
} // namespace

int main(int argument_count, char **argument_value) {
  if (argument_count != 2) {
    std::println("Usage: {} <result path>", argument_value[0]);
    return EXIT_FAILURE;
  }

  for (const auto &entry :
       std::filesystem::directory_iterator(std::filesystem::current_path())) {
    std::string filename{entry.path().filename().string()};
    if (entry.is_regular_file() &&
        (filename.starts_with("kalman_benchmark_predict") ||
         filename.starts_with("kalman_benchmark_update"))) {
      // std::string command{"./" + filename};
      if (int status{std::system(("./" + filename).c_str())}; status != 0) {
        std::println("Program: {} failed with code: {}", filename, status);
      }
    }
  }

  // Load the file in memory.
  for (std::string filename : {"predict.csv", "update.csv"}) {
    std::filesystem::path file_path{argument_value[1]};
    file_path /= filename;
    std::ifstream input(file_path);
    if (!input) {
      std::println("Failed to open file_path file: {}", argument_value[1]);
      return EXIT_FAILURE;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
      lines.push_back(line);
    }
    input.close();

    // Sort the content.
    std::sort(lines.begin(), lines.end());

    // Deduplicate the results.
    auto end{std::unique(lines.begin(), lines.end(),
                         [](const std::string &a, const std::string &b) {
                           return key_of(a) == key_of(b);
                         })};

    lines.erase(end, lines.end());

    // Atomically write the file in place.
    std::filesystem::path temporary{file_path.string() + ".tmp"};
    {
      std::ofstream output(temporary, std::ios::trunc);
      if (!output) {
        std::println("Failed to open temporary file: {}", temporary.string());
        return EXIT_FAILURE;
      }

      for (auto &&line : lines) {
        output << line << '\n';
      }
    }

    std::error_code ec;
    std::filesystem::rename(temporary, file_path, ec);
    if (ec) {
      std::println("Failed to replace file: {}", ec.message());
      std::filesystem::remove(temporary);
      return EXIT_FAILURE;
    }
  }

  // Plot.

  return EXIT_SUCCESS;
}
