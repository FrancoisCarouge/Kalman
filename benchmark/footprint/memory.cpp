/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.4
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

#include <cstddef>

//! @brief Minimal `memcpy` and `memset` routines a freestanding compiler may
//! call, bound to their names at link time.
//!
//! @details Compiled without loop pattern distribution, they call nothing: the
//! leaf functions bounding the stack of the entry points.
extern "C" {
auto footprint_copy(void *destination, const void *source, std::size_t size)
    -> void * {
  auto *to{static_cast<unsigned char *>(destination)};
  const auto *from{static_cast<const unsigned char *>(source)};

  for (std::size_t position{0}; position < size; ++position) {
    to[position] = from[position];
  }

  return destination;
}

auto footprint_fill(void *destination, int value, std::size_t size) -> void * {
  auto *to{static_cast<unsigned char *>(destination)};

  for (std::size_t position{0}; position < size; ++position) {
    to[position] = static_cast<unsigned char>(value);
  }

  return destination;
}
}
