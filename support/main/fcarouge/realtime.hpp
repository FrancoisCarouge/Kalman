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

#ifndef FCAROUGE_REALTIME_HPP
#define FCAROUGE_REALTIME_HPP

//! @file
//! @brief Opt-out of the real-time verification.

namespace fcarouge {
//! @brief Opt-out token of the real-time verification.
//!
//! @details Every test, sample, and benchmark is verified real-time by default:
//! built with the real-time sanitizer (`-fsanitize=realtime`), a call to a
//! real-time unsafe function, such as a memory allocation, a lock, or a
//! blocking input-output operation, aborts the program. An entry point that
//! does not support the real-time verification explicitly opts out by
//! declaring a named token as its first statement. The verification of the
//! calling thread resumes when the token goes out of scope. Built without the
//! real-time sanitizer, the token has no effect.
//!
//! @code{.cpp}
//! [[maybe_unused]] const auto test{[] -> int {
//!   const not_realtime opt_out;
//!   // ...
//! }()};
//! @endcode
class not_realtime {
public:
  //! @brief Opts the calling thread out of the real-time verification.
  [[nodiscard]] not_realtime() { disable(); }

  //! @brief A token pairs one disable with one enable of its thread.
  //!
  //! @details The implicit copy, and the move falling back to it, would run
  //! the destructor without the constructor: a copied token would resume the
  //! verification while the original still opts out.
  //! @{
  not_realtime(const not_realtime &) = delete;
  not_realtime(not_realtime &&) = delete;
  auto operator=(const not_realtime &) -> not_realtime & = delete;
  auto operator=(not_realtime &&) -> not_realtime & = delete;
  //! @}

  //! @brief Resumes the real-time verification of the calling thread.
  ~not_realtime() { enable(); }

private:
  //! @brief Disables, enables the real-time verification of the calling
  //! thread, when built with the real-time sanitizer.
  //! @{
  static void disable();
  static void enable();
  //! @}
};
} // namespace fcarouge

#endif // FCAROUGE_REALTIME_HPP
