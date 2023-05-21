/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.2.0
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

#ifndef FCAROUGE_INTERNAL_FUNCTION_HPP
#define FCAROUGE_INTERNAL_FUNCTION_HPP

#include <memory>

namespace fcarouge::internal {
// Compile-time `std::function` partial drop-in.
template <typename Undefined> class function;

template <typename Result, typename... Arguments>
class function<Result(Arguments...)> {
public:
  template <typename Callable>
  constexpr explicit function(Callable callee)
      : storage{std::make_unique<implementation<Callable>>(callee)} {}

  template <typename Callable> function &operator=(Callable &&callee) {
    storage = std::make_unique<implementation<Callable>>(callee);
    return *this;
  }

  constexpr auto operator()(Arguments... arguments) const -> Result {
    return (*storage)(arguments...);
  }

private:
  struct interface {
    constexpr virtual auto operator()(Arguments...) -> Result = 0;
    constexpr virtual ~interface() = default;
  };

  template <typename Callable> struct implementation final : interface {
  public:
    constexpr explicit implementation(Callable callee) : memory{callee} {}
    constexpr auto operator()(Arguments... arguments) -> Result override {
      return memory(arguments...);
    }
    constexpr ~implementation() override = default;

  private:
    Callable memory{};
  };

  //! @todo Support optimized small storage alternatives?
  std::unique_ptr<interface> storage{};
};

template <typename Callable> struct function_traits {};

template <typename Result, typename Type, typename... Arguments>
struct function_traits<Result (Type::*)(Arguments...) const> {
  using type = Result(Arguments...);
};

template <typename Callable>
using function_traits_t =
    typename function_traits<decltype(&Callable::operator())>::type;

template <typename Callable>
function(Callable) -> function<function_traits_t<Callable>>;

template <typename Result, typename... Arguments>
function(Result(Arguments...)) -> function<Result(Arguments...)>;
} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_FUNCTION_HPP
