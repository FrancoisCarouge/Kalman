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

//! @file
//! @brief Real-time verification of the tests, samples, and benchmarks.
//!
//! @details Built with the real-time sanitizer (`-fsanitize=realtime`), every
//! program linking this object runs its main thread in a real-time context
//! from before its first default-priority static initializer to after its
//! last default-priority static destructor, as if its entry point were
//! `[[clang::nonblocking]]`. Built without the real-time sanitizer, it has no
//! effect.

#include "fcarouge/realtime.hpp"

#ifdef __has_feature
#if __has_feature(realtime_sanitizer)
#define FCAROUGE_REALTIME_SANITIZER
#endif
#endif

#ifdef FCAROUGE_REALTIME_SANITIZER
#include <sanitizer/rtsan_interface.h>

//! @brief Enters, exits a real-time context of the calling thread.
//!
//! @details The real-time sanitizer runtime functions the compiler calls around
//! a `[[clang::nonblocking]]` function. The runtime exports them without
//! declaring them in its public interface header.
//! @{
extern "C" void __rtsan_realtime_enter();
extern "C" void __rtsan_realtime_exit();
//! @}

namespace fcarouge {
namespace {
//! @brief Enters the real-time context ahead of the default-priority static
//! initializers running the tests, samples, and benchmarks.
[[gnu::constructor(101)]] void realtime_enter() { __rtsan_realtime_enter(); }

//! @brief Exits the real-time context after the default-priority static
//! destructors.
[[gnu::destructor(101)]] void realtime_exit() { __rtsan_realtime_exit(); }
} // namespace

void not_realtime::disable() { __rtsan_disable(); }

void not_realtime::enable() { __rtsan_enable(); }
} // namespace fcarouge
#else
namespace fcarouge {
void not_realtime::disable() {}

void not_realtime::enable() {}
} // namespace fcarouge
#endif
