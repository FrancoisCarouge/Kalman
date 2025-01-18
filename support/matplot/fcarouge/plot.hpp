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

#ifndef FCAROUGE_PLOT_HPP
#define FCAROUGE_PLOT_HPP

#include "fcarouge/kalman_internal/utility.hpp"

#include <matplot/matplot.h>

#include <algorithm>
#include <cctype>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace fcarouge {
namespace kalman_internal {
template <typename Filter> class plotter : public Filter {
public:
  explicit plotter(Filter &&filter, const std::string &plot_name);
  ~plotter();
  void predict(const auto &...arguments);
  void update(const auto &...arguments);

private:
  using serie = std::vector<double>;

  void plot2(const std::string &characteristic_x,
             const std::string &characteristic_y, const serie &serie_x,
             const serie &serie_y);

  std::vector<Filter> history;
  std::string name;
};

template <typename Filter>
plotter<Filter>::plotter(Filter &&filter, const std::string &plot_name)
    : Filter{std::forward<Filter>(filter)},
      history{{static_cast<Filter &>(*this)}}, name{plot_name} {}

template <typename Filter> plotter<Filter>::~plotter() {
  std::unordered_map<std::string, serie> series{
      {"Index", matplot::linspace(0, history.size() - 1, history.size())}};

  if constexpr (has_state_method<Filter>) {
    series["State X"] =
        history |
        std::views::transform([](const Filter &filter) { return filter.x(); }) |
        std::ranges::to<serie>();
  }

  if constexpr (has_output_method<Filter>) {
    series["Measurement Z"] =
        history |
        std::views::transform([](const Filter &filter) { return filter.z(); }) |
        std::ranges::to<serie>();
  }

  if constexpr (has_input_method<Filter>) {
    series["Control U"] =
        history |
        std::views::transform([](const Filter &filter) { return filter.u(); }) |
        std::ranges::to<serie>();
  }

  if constexpr (has_estimate_uncertainty_method<Filter>) {
    series["Estimate Uncertainty P"] =
        history |
        std::views::transform([](const Filter &filter) { return filter.p(); }) |
        std::ranges::to<serie>();
  }

  if constexpr (has_process_uncertainty_method<Filter>) {
    series["Process Uncertainty Q"] =
        history |
        std::views::transform([](const Filter &filter) { return filter.q(); }) |
        std::ranges::to<serie>();
  }

  if constexpr (has_output_uncertainty_method<Filter>) {
    series["Output Uncertainty R"] =
        history |
        std::views::transform([](const Filter &filter) { return filter.r(); }) |
        std::ranges::to<serie>();
  }

  if constexpr (has_state_transition<Filter>) {
    series["State Transition F"] =
        history |
        std::views::transform([](const Filter &filter) { return filter.f(); }) |
        std::ranges::to<serie>();
  }

  if constexpr (has_output_model<Filter>) {
    series["Observation Transition H"] =
        history |
        std::views::transform([](const Filter &filter) { return filter.h(); }) |
        std::ranges::to<serie>();
  }

  if constexpr (has_input_control<Filter>) {
    series["Control Transition G"] =
        history |
        std::views::transform([](const Filter &filter) { return filter.g(); }) |
        std::ranges::to<serie>();
  }

  if constexpr (has_gain<Filter>) {
    series["Gain K"] =
        history |
        std::views::transform([](const Filter &filter) { return filter.k(); }) |
        std::ranges::to<serie>();
  }

  if constexpr (has_innovation_method<Filter>) {
    series["Innovation Y"] =
        history |
        std::views::transform([](const Filter &filter) { return filter.y(); }) |
        std::ranges::to<serie>();
  }

  if constexpr (has_innovation_uncertainty_method<Filter>) {
    series["Innovation Uncertainty S"] =
        history |
        std::views::transform([](const Filter &filter) { return filter.s(); }) |
        std::ranges::to<serie>();
  }

  for (auto [characteristic_x, serie_x] : series) {
    for (auto [characteristic_y, serie_y] : series) {
      plot2(characteristic_x, characteristic_y, serie_x, serie_y);
    }
  }
}

template <typename Filter>
void plotter<Filter>::predict(const auto &...arguments) {
  Filter::predict(arguments...);
  history.emplace_back(static_cast<Filter &>(*this));
}

template <typename Filter>
void plotter<Filter>::update(const auto &...arguments) {
  Filter::update(arguments...);
  history.emplace_back(static_cast<Filter &>(*this));
}

template <typename Filter>
void plotter<Filter>::plot2(const std::string &characteristic_x,
                            const std::string &characteristic_y,
                            const serie &serie_x, const serie &serie_y) {
  std::string title{
      name | std::ranges::views::transform([](char c) {
        return static_cast<char>(std::toupper(c));
      }) |
      std::views::transform([](char c) { return (c == '_') ? ' ' : c; }) |
      std::ranges::to<std::string>()};

  std::string subtitle{
      (characteristic_y + " over " + characteristic_x) |
      std::ranges::views::transform(
          [](char c) { return static_cast<char>(std::toupper(c)); }) |
      std::views::transform([](char c) { return (c == '_') ? ' ' : c; }) |
      std::ranges::to<std::string>()};

  std::string filename{
      (name + "_" + characteristic_y + "_" + characteristic_x) |
      std::ranges::views::transform(
          [](char c) { return static_cast<char>(std::tolower(c)); }) |
      std::views::transform([](char c) { return (c == ' ') ? '_' : c; }) |
      std::ranges::to<std::string>()};

  auto figure{matplot::figure(true)};
  auto axes{figure->current_axes()};
  axes->title(title + " - " + subtitle);
  axes->xlabel(characteristic_x);
  axes->ylabel(characteristic_y);
  auto plot{axes->plot(serie_x, serie_y, "-x")};
  plot->line_width(3);
  figure->save("plot/" + filename + ".png");
}
} // namespace kalman_internal

struct plotter {
  std::string plot_name{"Filter Plot Title"};
  [[nodiscard]] constexpr plotter
  operator()(const std::string &plot_name) const {
    return {plot_name};
  }
};

template <typename Filter>
[[nodiscard]] constexpr auto operator|(Filter &&filter,
                                       const plotter &decorator) {
  return kalman_internal::plotter<Filter>(std::forward<Filter>(filter),
                                          decorator.plot_name);
}

//! @name Decorators
//! @{

//! @brief Filter decorator to plot filter characteristics history.
plotter plot;

//! @}

} // namespace fcarouge

#endif // FCAROUGE_PLOT_HPP
