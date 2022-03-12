/*_____  _____   ____       _ ______ _____ _______
 |  __ \|  __ \ / __ \     | |  ____/ ____|__   __|
 | |__) | |__) | |  | |    | | |__ | |       | |
 |  ___/|  _  /| |  | |_   | |  __|| |       | |
 | |    | | \ \| |__| | |__| | |___| |____   | |
 |_|    |_|  \_\\____/ \____/|______\_____|  |_|

Project template for C++
Version 0.1.0
https://github.com/FrancoisCarouge/Project

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

#include <cassert>

#include "fcarouge/project.hpp"

namespace fcarouge::test
{
namespace
{
//! @test Verify the code.
[[maybe_unused]] constexpr auto ctest = []() {
  static_assert(true, "Compile-time test.");
  assert(true && "Run-time test.");

  return 0;
}();

} // namespace
} // namespace fcarouge::test
