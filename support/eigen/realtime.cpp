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

#ifdef __has_feature
#if __has_feature(realtime_sanitizer)
#define FCAROUGE_REALTIME_SANITIZER
#endif
#endif

#ifdef FCAROUGE_REALTIME_SANITIZER
#include <Eigen/Core>

namespace fcarouge::eigen {
namespace {
//! @brief Initializes the Eigen state lazily initialized on first use, ahead of
//! the real-time context entered at the next constructor priority.
//!
//! @details Eigen sizes the blocking of its matrix kernels, for example the
//! triangular solve of a division with a multi-column right-hand side, from
//! CPU cache sizes held in a function-local static variable. Its thread-safe
//! initialization on first use takes a lock, a real-time unsafe call. A
//! real-time application initializes it before entering its real-time
//! context, as done here for every test, sample, and benchmark of an Eigen
//! backend.
[[gnu::constructor(101)]] void realtime_prepare() {
  static_cast<void>(Eigen::l1CacheSize());
}
} // namespace
} // namespace fcarouge::eigen
#endif
