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

#include "fcarouge/kalman.hpp"
#include "fcarouge/linalg.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <numbers>
#include <tuple>
#include <utility>

#include <mp-units/math.h>
#include <mp-units/systems/isq/mechanics.h>
#include <mp-units/systems/isq/space_and_time.h>
#include <mp-units/systems/isq/thermodynamics.h>
#include <mp-units/systems/si.h>

namespace fcarouge::sample {
namespace {
//! @name The Atmospheric Quantities
//! @brief The quantity kinds of the estimated fields. The wind components are
//! kinds of their own: an eastward wind is never added to, nor substituted for,
//! a northward wind. The three mixing ratios share a unit, kilogram per
//! kilogram, but not a kind: humidity is never added to ozone.
//! @{
constexpr struct eastward_wind
    : mp_units::quantity_spec<mp_units::isq::velocity, mp_units::subkind> {
} eastward_wind;
constexpr struct northward_wind
    : mp_units::quantity_spec<mp_units::isq::velocity, mp_units::subkind> {
} northward_wind;
constexpr struct specific_humidity
    : mp_units::quantity_spec<mp_units::dimensionless, mp_units::subkind> {
} specific_humidity;
constexpr struct ozone_mixing_ratio
    : mp_units::quantity_spec<mp_units::dimensionless, mp_units::subkind> {
} ozone_mixing_ratio;
constexpr struct cloud_water_mixing_ratio
    : mp_units::quantity_spec<mp_units::dimensionless, mp_units::subkind> {
} cloud_water_mixing_ratio;
//! @}

using mp_units::delta;
using mp_units::one;
using mp_units::quantity;
using mp_units::si::unit_symbols::d;
using mp_units::si::unit_symbols::h;
using mp_units::si::unit_symbols::K;
using mp_units::si::unit_symbols::kg;
using mp_units::si::unit_symbols::km;
using mp_units::si::unit_symbols::m;
using mp_units::si::unit_symbols::m3;
using mp_units::si::unit_symbols::Pa;
using mp_units::si::unit_symbols::s;

using temperature = quantity<mp_units::isq::thermodynamic_temperature[K]>;
using eastward_velocity = quantity<eastward_wind[m / s]>;
using northward_velocity = quantity<northward_wind[m / s]>;
using humidity = quantity<specific_humidity[kg / kg]>;
using ozone = quantity<ozone_mixing_ratio[kg / kg]>;
using cloud_water = quantity<cloud_water_mixing_ratio[kg / kg]>;
using pressure = quantity<mp_units::isq::pressure[Pa]>;
using duration = quantity<mp_units::isq::duration[s]>;

//! @name The Grid
//! @brief Four longitudes, 90° apart and periodic, at 45°N and 45°S: eight
//! columns, each of two levels, 850 hPa and 300 hPa.
//! @{
inline constexpr std::size_t longitudes{4};
inline constexpr std::size_t columns{2 * longitudes};
inline constexpr std::size_t levels{2};
//! @}

//! @name The Variables
//! @brief The variables of a column, at each level, then its surface pressure.
//! @{
inline constexpr std::size_t air_temperature{0};
inline constexpr std::size_t eastward{1};
inline constexpr std::size_t northward{2};
inline constexpr std::size_t humidity_ratio{3};
inline constexpr std::size_t ozone_ratio{4};
inline constexpr std::size_t cloud_water_ratio{5};
inline constexpr std::size_t surface_pressure{6};
//! @}

//! @brief The state elements of a column: six variables at each level, and the
//! surface pressure.
inline constexpr std::size_t column_size{(surface_pressure * levels) + 1};
inline constexpr std::size_t state_size{columns * column_size};
inline constexpr std::size_t output_size{12};

//! @brief The index of a variable of a column at a level in the state: column
//! after column, the levels of a variable together.
inline constexpr auto index_of{[](std::size_t column, std::size_t field,
                                  std::size_t level) -> std::size_t {
  return (column * column_size) + (field * levels) + level;
}};

//! @brief The index of the surface pressure of a column in the state.
inline constexpr auto pressure_of{[](std::size_t column) -> std::size_t {
  return (column * column_size) + (surface_pressure * levels);
}};

//! @brief A state element: its column, variable, and level.
struct element {
  std::size_t column;
  std::size_t field;
  std::size_t level;
};

//! @brief The element of an index of the state.
inline constexpr auto element_of{[](std::size_t index) -> element {
  const std::size_t offset{index % column_size};

  return {.column = index / column_size,
          .field = offset / levels,
          .level = offset % levels};
}};

//! @brief The hemisphere of a column: 0 at 45°N, 1 at 45°S.
inline constexpr auto hemisphere_of{
    [](std::size_t column) -> std::size_t { return column / longitudes; }};

//! @brief The longitude of a column: 0 at 0°, 1 at 90°E, 2 at 180°, 3 at 90°W.
inline constexpr auto longitude_of{
    [](std::size_t column) -> std::size_t { return column % longitudes; }};

//! @brief The neighbouring column to the west.
inline constexpr auto west_of{[](std::size_t column) -> std::size_t {
  return (hemisphere_of(column) * longitudes) +
         ((longitude_of(column) + longitudes - 1) % longitudes);
}};

//! @brief The neighbouring column to the east.
inline constexpr auto east_of{[](std::size_t column) -> std::size_t {
  return (hemisphere_of(column) * longitudes) +
         ((longitude_of(column) + 1) % longitudes);
}};

//! @brief The column of the same longitude in the other hemisphere.
inline constexpr auto across_of{[](std::size_t column) -> std::size_t {
  return ((1 - hemisphere_of(column)) * longitudes) + longitude_of(column);
}};

//! @brief Whether the variable is carried by the winds: the temperature and
//! the three mixing ratios.
inline constexpr auto is_tracer{[](std::size_t field) -> bool {
  return field == air_temperature || field == humidity_ratio ||
         field == ozone_ratio || field == cloud_water_ratio;
}};

//! @brief Whether the next value of a state element depends on the current
//! value of another, through the model below: a tracer on itself, on its winds,
//! and on its upwind neighbours; a wind on itself and on the surface pressure
//! gradient; the surface pressure on itself and on the convergence of the lower
//! winds.
inline constexpr auto depends{[](std::size_t row, std::size_t column) -> bool {
  const element lhs{element_of(row)};
  const element rhs{element_of(column)};
  const std::size_t here{lhs.column};

  if (row == column) {
    return true;
  }

  if (is_tracer(lhs.field)) {
    return (rhs.column == here && rhs.level == lhs.level &&
            (rhs.field == eastward || rhs.field == northward)) ||
           (rhs.field == lhs.field && rhs.level == lhs.level &&
            (rhs.column == west_of(here) || rhs.column == east_of(here) ||
             rhs.column == across_of(here)));
  }

  if (lhs.field == eastward) {
    return column == pressure_of(west_of(here)) ||
           column == pressure_of(east_of(here));
  }

  if (lhs.field == northward) {
    return column == pressure_of(here) ||
           column == pressure_of(across_of(here));
  }

  return column == index_of(west_of(here), eastward, 0) ||
         column == index_of(east_of(here), eastward, 0);
}};

//! @brief The quantity types of the state elements of a column, in order.
using column_quantities =
    std::tuple<temperature, temperature, eastward_velocity, eastward_velocity,
               northward_velocity, northward_velocity, humidity, humidity,
               ozone, ozone, cloud_water, cloud_water, pressure>;

template <typename Tuple, typename Sequence> struct repeated;

template <typename Tuple, std::size_t... Index>
struct repeated<Tuple, std::index_sequence<Index...>> {
  template <std::size_t> using copy = Tuple;

  using type = decltype(std::tuple_cat(std::declval<copy<Index>>()...));
};

template <typename Tuple> struct reciprocals;

template <typename... Types> struct reciprocals<std::tuple<Types...>> {
  using type = std::tuple<quantity<mp_units::one / Types::reference>...>;
};

template <typename Tuple> struct vector_of;

template <typename... Types> struct vector_of<std::tuple<Types...>> {
  using type = column_vector<double, Types...>;
};

//! @brief The state elements: the row and column indexes of a covariance of
//! the state.
using state_indexes =
    repeated<column_quantities, std::make_index_sequence<columns>>::type;
//! @brief The output elements: a sounding of the temperature, winds, and
//! humidity of both levels of a column; the surface pressure of two stations;
//! and the column ozone and cloud water of two satellite retrievals.
using output_indexes =
    std::tuple<temperature, temperature, eastward_velocity, eastward_velocity,
               northward_velocity, northward_velocity, humidity, humidity,
               pressure, pressure, ozone, cloud_water>;
//! @brief The reciprocals of the state elements: the column indexes of a
//! derivative by the state.
using reciprocal_state_indexes = reciprocals<state_indexes>::type;

//! @brief The state: the temperature [K], eastward and northward winds
//! [m.s^-1], specific humidity, ozone, and cloud water [kg.kg^-1] of both
//! levels, and the surface pressure [Pa], of each column.
using atmosphere = vector_of<state_indexes>::type;
//! @brief The output, of the observations described above.
using observations = vector_of<output_indexes>::type;

//! @name The Observation Network
//! @brief The column of the sounding, of the two surface pressure stations,
//! and of the ozone and cloud water retrievals.
//! @{
inline constexpr std::size_t sounding_column{1};
inline constexpr std::size_t first_station_column{0};
inline constexpr std::size_t second_station_column{6};
inline constexpr std::size_t ozone_column{5};
inline constexpr std::size_t cloud_water_column{3};
//! @}

//! @brief The weight of the mass of the lower layer, 1000 hPa to 600 hPa, and
//! of the upper layer, 600 hPa to 150 hPa, in a column mean.
inline constexpr double lower_weight{400. / 850.};
inline constexpr double upper_weight{450. / 850.};

//! @brief Whether an output observes a state element: the sounding, the
//! stations, and the layer-weighted column means of the retrievals.
inline constexpr auto observes{[](std::size_t row, std::size_t column) -> bool {
  if (row < 8) {
    return column == (sounding_column * column_size) + row;
  }

  switch (row) {
  case 8:
    return column == pressure_of(first_station_column);
  case 9:
    return column == pressure_of(second_station_column);
  case 10:
    return column == index_of(ozone_column, ozone_ratio, 0) ||
           column == index_of(ozone_column, ozone_ratio, 1);
  default:
    return column == index_of(cloud_water_column, cloud_water_ratio, 0) ||
           column == index_of(cloud_water_column, cloud_water_ratio, 1);
  }
}};

//! @brief The covariance localization: a column correlates with itself and its
//! two neighbouring columns of the same hemisphere, 60° away. The columns 90°
//! away or more, at the opposite longitude or in the other hemisphere, are
//! beyond the cutoff: 4,056 stored elements of 10,816.
inline constexpr sparse::pattern localization_pattern{
    sparse::shape<state_size, state_size>,
    [](std::size_t row, std::size_t column) -> bool {
      const std::size_t lhs{row / column_size};
      const std::size_t rhs{column / column_size};

      return hemisphere_of(lhs) == hemisphere_of(rhs) &&
             ((longitude_of(lhs) + longitudes - longitude_of(rhs)) %
              longitudes) != 2;
    }};
//! @brief The derivatives of the state transition F, of the dependencies of
//! the model: 504 stored elements of 10,816.
inline constexpr sparse::pattern transition_pattern{
    sparse::shape<state_size, state_size>,
    [](std::size_t row, std::size_t column) -> bool {
      return depends(row, column);
    }};
//! @brief The derivatives of the output model H: the observed elements, and
//! the diagonal holding its default value, the identity, before the first
//! update: 26 stored elements of 1,248.
inline constexpr sparse::pattern measurement_pattern{
    sparse::shape<output_size, state_size>,
    [](std::size_t row, std::size_t column) -> bool {
      return row == column || observes(row, column);
    }};

static_assert(localization_pattern.count == 4'056 &&
              transition_pattern.count == 504 &&
              measurement_pattern.count == 26);

//! @brief The state transition F: the derivatives of the state by the state.
using transition_jacobian =
    typed_sparse_matrix<double, transition_pattern, state_indexes,
                        reciprocal_state_indexes>;
//! @brief The output model H: the derivatives of the output by the state.
using measurement_jacobian =
    typed_sparse_matrix<double, measurement_pattern, output_indexes,
                        reciprocal_state_indexes>;
//! @brief The localized estimate uncertainty P: the covariances of the state
//! within the cutoff.
using localized_covariance = typed_sparse_matrix<double, localization_pattern,
                                                 state_indexes, state_indexes>;
//! @brief A covariance of the state, without structural zeros.
using state_covariance = matrix<double, state_indexes, state_indexes>;
//! @brief A covariance of the output, without structural zeros.
using output_covariance = matrix<double, output_indexes, output_indexes>;
//! @brief The taper of the localization, of dimensionless elements.
using taper_matrix = typed_sparse_matrix<double, localization_pattern>;

//! @brief The taper of the localization: one within a column, 0.4 between
//! neighbouring columns. Positive semi-definite: in each hemisphere, it is the
//! Kronecker product of the correlations of the four columns around their
//! ring, of eigenvalues 1.8, 1, 1, and 0.2, by the matrix of ones of the
//! thirteen elements of a column, of eigenvalues 13 and twelve zeros. Its
//! diagonal is of ones: by the Schur product theorem, its product by a
//! positive definite estimate uncertainty is positive definite.
struct taper {
  static constexpr taper_matrix value{[] -> taper_matrix {
    taper_matrix result;

    for (std::size_t index{0}; index < localization_pattern.count; ++index) {
      const sparse::position position{localization_pattern.positions.at(index)};

      result.data().values.at(index) =
          position.row / column_size == position.column / column_size ? 1.
                                                                      : 0.4;
    }

    return result;
  }()};
};

//! @brief The quantity type of a state element.
template <std::size_t Index>
using element_type = std::tuple_element_t<Index, state_indexes>;

//! @brief The quantity type of the rate of change of a quantity.
template <typename Quantity>
using rate_of = quantity<(Quantity::quantity_spec /
                          mp_units::isq::duration)[Quantity::unit / s]>;

//! @brief The quantity type of a dimensionless ratio.
using ratio = quantity<mp_units::dimensionless[one]>;

//! @name The Model
//! @brief The physical constants of the model.
//! @{
//! The Earth's mean radius.
inline constexpr quantity<mp_units::isq::length[m]> earth_radius{6'371. * km};
//! The distance between neighbouring longitudes at 45°, and between the two
//! latitudes.
inline constexpr quantity<mp_units::isq::length[m]> zonal_spacing{
    earth_radius * std::numbers::sqrt2 / 2. * std::numbers::pi / 2.};
inline constexpr quantity<mp_units::isq::length[m]> meridional_spacing{
    earth_radius * std::numbers::pi / 2.};
//! The air density of each level.
inline constexpr std::array<quantity<mp_units::isq::density[kg / m3]>, levels>
    air_density{1.1 * kg / m3, 0.45 * kg / m3};
//! The mean surface pressure.
inline constexpr pressure mean_pressure{101'325. * Pa};
//! The pressure thickness of the boundary layer, whose wind convergence changes
//! the surface pressure.
inline constexpr pressure boundary_layer{10'000. * Pa};
//! The relaxation times of the tracers, the winds, and the surface pressure.
inline constexpr duration tracer_relaxation{10. * d};
inline constexpr duration wind_relaxation{4. * d};
inline constexpr duration pressure_relaxation{2. * d};
//! @}

//! @brief The longitudinal wave of the reference climate at a column: as over
//! continents and oceans, differing between the hemispheres.
inline constexpr auto wave_of{[](std::size_t column) -> double {
  return std::cos(
      (std::numbers::pi / 2. * static_cast<double>(longitude_of(column))) +
      (std::numbers::pi / 3. * static_cast<double>(hemisphere_of(column))));
}};

//! @brief The reference value of a state element, which the model relaxes it
//! toward: the climate the winds carry the fields away from.
template <std::size_t Index>
inline constexpr auto reference_of{[] -> element_type<Index> {
  constexpr element at{element_of(Index)};
  constexpr bool lower{at.level == 0};
  const double wave{wave_of(at.column)};

  if constexpr (at.field == air_temperature) {
    return delta<K>(lower ? 283. : 228.) + (wave * delta<K>(5.));
  } else if constexpr (at.field == eastward) {
    return (lower ? 6. : 22.) * eastward_wind[m / s];
  } else if constexpr (at.field == northward) {
    return northward_velocity::zero();
  } else if constexpr (at.field == humidity_ratio) {
    return (lower ? 6e-3 : 3e-4) * (1. + (0.3 * wave)) *
           specific_humidity[kg / kg];
  } else if constexpr (at.field == ozone_ratio) {
    return (lower ? 6e-8 : 4e-6) * (1. + (0.2 * wave)) *
           ozone_mixing_ratio[kg / kg];
  } else if constexpr (at.field == cloud_water_ratio) {
    return (lower ? 8e-5 : 1e-5) * (1. + (0.5 * wave)) *
           cloud_water_mixing_ratio[kg / kg];
  } else {
    return mean_pressure + (wave * 800. * Pa);
  }
}};

//! @brief The standard deviation of the model error of a state element over an
//! hour, scaled.
template <std::size_t Index>
inline constexpr auto deviation_of{[](double scale) -> element_type<Index> {
  constexpr std::size_t field{element_of(Index).field};

  if constexpr (field == air_temperature) {
    return scale * delta<K>(0.1);
  } else if constexpr (field == eastward) {
    return scale * 0.2 * eastward_wind[m / s];
  } else if constexpr (field == northward) {
    return scale * 0.2 * northward_wind[m / s];
  } else if constexpr (field == humidity_ratio) {
    return scale * 2e-5 * specific_humidity[kg / kg];
  } else if constexpr (field == ozone_ratio) {
    return scale * 1e-8 * ozone_mixing_ratio[kg / kg];
  } else if constexpr (field == cloud_water_ratio) {
    return scale * 2e-6 * cloud_water_mixing_ratio[kg / kg];
  } else {
    return scale * 10. * Pa;
  }
}};

//! @brief The rate of change of a state element: the upwind advection of a
//! tracer by the winds of its level along its latitude, and across the equator
//! from the other hemisphere, the acceleration of a wind by the surface
//! pressure gradient, and the change of the surface pressure by the
//! convergence of the lower winds, each relaxed toward its reference.
template <std::size_t Index>
inline constexpr auto tendency{
    [](const atmosphere &x) -> rate_of<element_type<Index>> {
      using rate = rate_of<element_type<Index>>;

      constexpr element at{element_of(Index)};
      constexpr std::size_t here{at.column};
      constexpr std::size_t west{west_of(here)};
      constexpr std::size_t east{east_of(here)};
      constexpr std::size_t across{across_of(here)};
      constexpr bool north{hemisphere_of(here) == 0};

      const element_type<Index> relaxed{x.at<Index>() - reference_of<Index>()};

      if constexpr (is_tracer(at.field)) {
        const eastward_velocity u{x.at<index_of(here, eastward, at.level)>()};
        const northward_velocity v{x.at<index_of(here, northward, at.level)>()};
        const northward_velocity inflow{
            std::max(north ? v : -v, northward_velocity::zero())};
        const element_type<Index> field{x.at<Index>()};
        const element_type<Index> upwind{
            u > eastward_velocity::zero()
                ? field - x.at<index_of(west, at.field, at.level)>()
                : x.at<index_of(east, at.field, at.level)>() - field};

        return rate{-u * upwind / zonal_spacing} -
               rate{inflow *
                    (field - x.at<index_of(across, at.field, at.level)>()) /
                    meridional_spacing} -
               rate{relaxed / tracer_relaxation};
      } else if constexpr (at.field == eastward) {
        return rate{-(x.at<pressure_of(east)>() - x.at<pressure_of(west)>()) /
                    (2. * zonal_spacing * std::get<at.level>(air_density))} -
               rate{relaxed / wind_relaxation};
      } else if constexpr (at.field == northward) {
        const pressure southward_difference{
            north ? x.at<pressure_of(here)>() - x.at<pressure_of(across)>()
                  : x.at<pressure_of(across)>() - x.at<pressure_of(here)>()};

        return rate{-southward_difference /
                    (meridional_spacing * std::get<at.level>(air_density))} -
               rate{relaxed / wind_relaxation};
      } else {
        return rate{-boundary_layer *
                    (x.at<index_of(east, eastward, 0)>() -
                     x.at<index_of(west, eastward, 0)>()) /
                    (2. * zonal_spacing)} -
               rate{relaxed / pressure_relaxation};
      }
    }};

//! @brief The state after a time step: an explicit Euler step of the model.
inline constexpr auto forecast{[](const atmosphere &x,
                                  const duration &step) -> atmosphere {
  return [&x, &step]<std::size_t... Index>(
             std::index_sequence<Index...>) -> atmosphere {
    return atmosphere{
        element_type<Index>{x.at<Index>() + (step * tendency<Index>(x))}...};
  }(std::make_index_sequence<state_size>{});
}};

//! @brief The state transition F of a time step at a position of its pattern:
//! the derivative of the next state element of the row by the current state
//! element of the column.
template <std::size_t Position>
inline constexpr auto derivative{[](const atmosphere &x,
                                    const duration &step) -> auto {
  constexpr std::size_t row{transition_pattern.positions.at(Position).row};
  constexpr std::size_t column{
      transition_pattern.positions.at(Position).column};
  constexpr element at{element_of(row)};
  constexpr std::size_t here{at.column};
  constexpr std::size_t west{west_of(here)};
  constexpr std::size_t east{east_of(here)};
  constexpr std::size_t across{across_of(here)};
  constexpr bool north{hemisphere_of(here) == 0};

  using type = transition_jacobian::element<row, column>;

  if constexpr (is_tracer(at.field)) {
    const eastward_velocity u{x.at<index_of(here, eastward, at.level)>()};
    const northward_velocity v{x.at<index_of(here, northward, at.level)>()};
    const bool westerly{u > eastward_velocity::zero()};
    const eastward_velocity from_west{westerly ? u : eastward_velocity::zero()};
    const eastward_velocity from_east{westerly ? eastward_velocity::zero() : u};
    const northward_velocity poleward{north ? v : -v};
    const bool inflowing{poleward > northward_velocity::zero()};
    const northward_velocity inflow{inflowing ? poleward
                                              : northward_velocity::zero()};
    const element_type<row> field{x.at<row>()};
    const element_type<row> from_across{
        inflowing ? field - x.at<index_of(across, at.field, at.level)>()
                  : element_type<row>::zero()};

    if constexpr (column == row) {
      return type{(1. * one) - ratio{step * abs(u) / zonal_spacing} -
                  ratio{step * inflow / meridional_spacing} -
                  ratio{step / tracer_relaxation}};
    } else if constexpr (column == index_of(west, at.field, at.level)) {
      return type{step * from_west / zonal_spacing};
    } else if constexpr (column == index_of(east, at.field, at.level)) {
      return type{-step * from_east / zonal_spacing};
    } else if constexpr (column == index_of(across, at.field, at.level)) {
      return type{step * inflow / meridional_spacing};
    } else if constexpr (column == index_of(here, eastward, at.level)) {
      return type{-step *
                  (westerly
                       ? field - x.at<index_of(west, at.field, at.level)>()
                       : x.at<index_of(east, at.field, at.level)>() - field) /
                  zonal_spacing};
    } else {
      return type{(north ? -step : step) * from_across / meridional_spacing};
    }
  } else if constexpr (at.field == eastward) {
    const auto gradient{step /
                        (2. * zonal_spacing * std::get<at.level>(air_density))};

    if constexpr (column == row) {
      return type{(1. * one) - (step / wind_relaxation)};
    } else if constexpr (column == pressure_of(west)) {
      return type{gradient};
    } else {
      return type{-gradient};
    }
  } else if constexpr (at.field == northward) {
    const auto gradient{step /
                        (meridional_spacing * std::get<at.level>(air_density))};

    if constexpr (column == row) {
      return type{(1. * one) - (step / wind_relaxation)};
    } else if constexpr ((column == pressure_of(across)) == north) {
      return type{gradient};
    } else {
      return type{-gradient};
    }
  } else {
    const auto convergence{step * boundary_layer / (2. * zonal_spacing)};

    if constexpr (column == row) {
      return type{(1. * one) - (step / pressure_relaxation)};
    } else if constexpr (column == index_of(west, eastward, 0)) {
      return type{convergence};
    } else {
      return type{-convergence};
    }
  }
}};

//! @brief The state transition F of a time step: the derivatives of the next
//! state by the current one, at the positions of its pattern.
inline constexpr auto jacobian_of{
    [](const atmosphere &x, const duration &step) -> transition_jacobian {
      transition_jacobian result;

      [&]<std::size_t... Position>(std::index_sequence<Position...>) -> void {
        (result.at<transition_pattern.positions.at(Position).row,
                   transition_pattern.positions.at(Position).column>(
             derivative<Position>(x, step)),
         ...);
      }(std::make_index_sequence<transition_pattern.count>{});

      return result;
    }};

//! @brief The observations of a state: the sounding, the stations, and the
//! layer-weighted column means of ozone and cloud water.
inline constexpr auto observe{[](const atmosphere &x) -> observations {
  constexpr std::size_t sounding{sounding_column * column_size};

  return observations{
      x.at<sounding>(),
      x.at<sounding + 1>(),
      x.at<sounding + 2>(),
      x.at<sounding + 3>(),
      x.at<sounding + 4>(),
      x.at<sounding + 5>(),
      x.at<sounding + 6>(),
      x.at<sounding + 7>(),
      x.at<pressure_of(first_station_column)>(),
      x.at<pressure_of(second_station_column)>(),
      (lower_weight * x.at<index_of(ozone_column, ozone_ratio, 0)>()) +
          (upper_weight * x.at<index_of(ozone_column, ozone_ratio, 1)>()),
      (lower_weight *
       x.at<index_of(cloud_water_column, cloud_water_ratio, 0)>()) +
          (upper_weight *
           x.at<index_of(cloud_water_column, cloud_water_ratio, 1)>())};
}};

//! @brief Sets an element of the output model H, of a dimensionless weight.
template <std::size_t Row, std::size_t Column>
inline constexpr auto weigh{
    [](measurement_jacobian &value, double weight) -> void {
      value.at<Row, Column>(
          measurement_jacobian::element<Row, Column>{weight * one});
    }};

//! @brief The output model H: the derivatives of the observations by the state,
//! constant.
inline constexpr auto measure{
    []([[maybe_unused]] const atmosphere &x) -> measurement_jacobian {
      constexpr std::size_t sounding{sounding_column * column_size};
      measurement_jacobian value;

      [&value]<std::size_t... Row>(std::index_sequence<Row...>) -> void {
        (weigh<Row, sounding + Row>(value, 1.), ...);
      }(std::make_index_sequence<8>{});
      weigh<8, pressure_of(first_station_column)>(value, 1.);
      weigh<9, pressure_of(second_station_column)>(value, 1.);
      weigh<10, index_of(ozone_column, ozone_ratio, 0)>(value, lower_weight);
      weigh<10, index_of(ozone_column, ozone_ratio, 1)>(value, upper_weight);
      weigh<11, index_of(cloud_water_column, cloud_water_ratio, 0)>(
          value, lower_weight);
      weigh<11, index_of(cloud_water_column, cloud_water_ratio, 1)>(
          value, upper_weight);

      return value;
    }};

//! @brief A covariance of the variances of each element, without correlations.
template <typename Covariance>
inline constexpr auto diagonal{[](const auto &deviation) -> Covariance {
  Covariance result;

  [&result,
   &deviation]<std::size_t... Index>(std::index_sequence<Index...>) -> void {
    (result.template at<Index, Index>(
         typename Covariance::template element<Index, Index>{
             pow<2>(deviation.template operator()<Index>())}),
     ...);
  }(std::make_index_sequence<Covariance::rows>{});

  return result;
}};

//! @brief The standard deviation of the noise of each output.
inline constexpr std::tuple<temperature, temperature, eastward_velocity,
                            eastward_velocity, northward_velocity,
                            northward_velocity, humidity, humidity, pressure,
                            pressure, ozone, cloud_water>
    output_deviations{delta<K>(0.5),
                      delta<K>(0.5),
                      1. * eastward_wind[m / s],
                      1. * eastward_wind[m / s],
                      1. * northward_wind[m / s],
                      1. * northward_wind[m / s],
                      5e-4 * specific_humidity[kg / kg],
                      5e-5 * specific_humidity[kg / kg],
                      50. * Pa,
                      50. * Pa,
                      1e-7 * ozone_mixing_ratio[kg / kg],
                      5e-6 * cloud_water_mixing_ratio[kg / kg]};

//! @name The Simulation
//! @brief The time step of the predictions, the period of the updates, and
//! the simulated period.
//! @{
inline constexpr duration time_step{1. * h};
inline constexpr duration cycle{6. * h};
inline constexpr duration period{10. * d};
//! @}

//! @brief A deterministic noise of zero mean and unit standard deviation.
const auto noise{[](std::size_t step, std::size_t index) -> double {
  return std::numbers::sqrt2 * std::sin((12.9898 * static_cast<double>(step)) +
                                        (78.233 * static_cast<double>(index)));
}};

//! @brief The global weather: the temperature, winds, humidity, ozone, cloud
//! water, and surface pressure of a coarse three-dimension global grid, an
//! extended filter of 104 states, of every vector and matrix of quantities,
//! and of the estimate uncertainty localized in a compile-time sparsity
//! pattern.
//!
//! @details The largest Kalman filters in use are the ensemble filters of
//! numerical weather prediction: 10⁸ to 10⁹ states of the atmosphere on a
//! global grid, millions of observations per cycle, and no control input. They
//! keep the problem tractable by localization: the correlations of the
//! estimate uncertainty between distant points of the grid are tapered to
//! zero, and never stored. This sample shows their structure at a scale that
//! fits in a sample.
//!
//! The grid has four longitudes, 90° apart, at 45°N and 45°S, and two levels,
//! 850 hPa and 300 hPa: eight columns, sixteen points. The state of each
//! column holds the temperature, the eastward and northward winds, the specific
//! humidity, the ozone and cloud water mixing ratios at each level, and the
//! surface pressure: thirteen elements, 104 in all. Each element is a quantity
//! of its own kind: adding the ozone to the humidity, or an eastward wind to a
//! northward one, does not compile, and neither does an observation of the
//! wrong quantity.
//!
//! The model carries the temperature and the mixing ratios by upwind advection
//! with the winds of their level, along the periodic latitude and across the
//! equator; accelerates the winds by the surface pressure gradient; changes the
//! surface pressure by the convergence of the lower winds; and relaxes every
//! field toward a reference climate, which varies with the longitude. The
//! winds multiply the advected fields: the model is nonlinear, so the filter
//! is extended, its state transition F the Jacobian of the model, of 504
//! stored elements of 10,816. The truth runs the same model from a different
//! initial state, perturbed by its model error, for ten days of one-hour
//! steps.
//!
//! Every six hours, the filter updates from twelve observations: a sounding of
//! the temperature, the winds, and the humidity of both levels of a column;
//! the surface pressure of two stations; and two satellite retrievals, of the
//! ozone and of the cloud water, each the mean of both levels of a column,
//! weighted by the mass of their layers. The output model H stores 26 elements
//! of 1,248:
//!
//! @image html ekf_104x12x0_global_weather_unit_layout.svg
//!
//! The estimate uncertainty P is localized: after each prediction and update,
//! the filter multiplies it element by element by a taper ρ, `P = ρ ∘ P`,
//! which keeps the correlations of a column with itself and with its two
//! neighbouring columns of the same hemisphere, 60° away, and drops those of
//! the columns 90° away or more. P is stored in the pattern of the taper:
//! 4,056 elements of 10,816. The taper is positive definite: by the Schur
//! product theorem, it keeps the estimate uncertainty positive definite. The
//! observations of a column correct its neighbours through their
//! correlations, and the model carries the corrections further.
//!
//! The filter starts from the climate, while the truth starts from a planetary
//! wave of the size of the uncertainty of that first estimate. In ten days,
//! the error of the 104 states, each in units of its hourly model error, falls
//! from 14 to 3, while the error of the climate grows to 26. The sounded
//! temperature is found within its noise from the first update, its
//! neighbouring column of the same hemisphere more slowly, and the surface
//! pressure of an unobserved column within days, from the winds of the
//! sounding:
//!
//! @image html ekf_104x12x0_global_weather_unit.svg
//!
//! The model, its Jacobian, and its constants are quantities throughout. Each
//! state element is computed by its compile-time index, of its own quantity
//! type: the rate of change of a temperature is in kelvin per second, and the
//! derivative of a temperature by an eastward wind in kelvin per metre per
//! second. The filter declares its state transition F and output model H of
//! sparsity patterns, which its callables compute exactly.
//!
//! @include{doc} ekf_104x12x0_global_weather_unit_benchmark.md
//!
//! @example ekf_104x12x0_global_weather_unit.cpp
[[maybe_unused]] const auto sample{[] -> int {
  // The root mean square of the errors of an estimate of the truth, each in
  // units of the model error of its element over a step.
  const auto normalized{
      [](const atmosphere &estimate, const atmosphere &truth) -> quantity<one> {
        return [&estimate, &truth]<std::size_t... Index>(
                   std::index_sequence<Index...>) -> quantity<one> {
          return sqrt((pow<2>((estimate.at<Index>() - truth.at<Index>()) /
                              deviation_of<Index>(1.)) +
                       ...) /
                      static_cast<double>(state_size));
        }(std::make_index_sequence<state_size>{});
      }};

  const atmosphere climate{
      []<std::size_t... Index>(std::index_sequence<Index...>) -> atmosphere {
        return atmosphere{reference_of<Index>()...};
      }(std::make_index_sequence<state_size>{})};

  kalman filter{
      // The state X: the reference climate.
      state{climate},
      // The output Z: the observations of the network.
      output<observations>,
      // The estimate uncertainty P: the uncertainty of the climate, localized.
      estimate_uncertainty{diagonal<localized_covariance>(
          []<std::size_t Index>() -> element_type<Index> {
            return deviation_of<Index>(20.);
          })},
      // The process uncertainty Q: the model error over a step.
      process_uncertainty{diagonal<state_covariance>(
          []<std::size_t Index>() -> element_type<Index> {
            return deviation_of<Index>(1.);
          })},
      // The output uncertainty R: the noise of each instrument.
      output_uncertainty{diagonal<output_covariance>(
          []<std::size_t Index>()
              -> std::tuple_element_t<Index, output_indexes> {
            return std::get<Index>(output_deviations);
          })},
      // The output model H:
      output_model{measure},
      // The state transition F, the Jacobian of the model over the step:
      state_transition{jacobian_of},
      // The transition f(X), the model over the step:
      transition{forecast},
      // The observation h(X):
      observation{observe},
      // The filter prediction uses the time step.
      prediction_types<duration>,
      // The state transition F and the output model H of sparsity patterns.
      state_transition_type<transition_jacobian>,
      output_model_type<measurement_jacobian>,
      // The localization of the estimate uncertainty P by its taper.
      localize<taper>};

  static_assert(std::same_as<decltype(filter)::estimate_uncertainty,
                             localized_covariance>,
                "The estimate uncertainty is stored in the pattern of the "
                "taper.");

  // The truth: the reference climate, displaced by a planetary wave of the
  // size of the uncertainty of the first estimate.
  atmosphere truth{[&climate]<std::size_t... Index>(
                       std::index_sequence<Index...>) -> atmosphere {
    return atmosphere{element_type<Index>{
        climate.at<Index>() +
        (std::cos(
             (std::numbers::pi / 2. *
              static_cast<double>(longitude_of(element_of(Index).column))) +
             (std::numbers::pi / 4. *
              static_cast<double>(hemisphere_of(element_of(Index).column))) +
             (0.7 * static_cast<double>(element_of(Index).field)) +
             (0.3 * static_cast<double>(element_of(Index).level))) *
         deviation_of<Index>(20.))}...};
  }(std::make_index_sequence<state_size>{})};

  const quantity<one> initial_error{normalized(filter.x(), truth)};

  duration until_update{cycle};
  std::size_t step{0};

  for (duration time{time_step}; time <= period; time += time_step) {
    ++step;
    truth = [&truth, step]<std::size_t... Index>(
                std::index_sequence<Index...>) -> atmosphere {
      const atmosphere forecasted{forecast(truth, time_step)};

      return atmosphere{element_type<Index>{
          forecasted.at<Index>() +
          (noise(step, Index) * deviation_of<Index>(1.))}...};
    }(std::make_index_sequence<state_size>{});

    filter.predict(time_step);

    // Every cycle, observe the truth, with the noise of each instrument.
    until_update -= time_step;

    if (until_update <= duration::zero()) {
      const observations observed{observe(truth)};

      filter.update([&observed, step]<std::size_t... Index>(
                        std::index_sequence<Index...>) -> observations {
        return observations{
            (observed.at<Index>() + (noise(step + 1000, Index) *
                                     std::get<Index>(output_deviations)))...};
      }(std::make_index_sequence<output_size>{}));
      until_update += cycle;
    }
  }

  constexpr std::size_t sounded{index_of(sounding_column, air_temperature, 0)};

  assert(normalized(filter.x(), truth) < initial_error / 3. &&
         "The atmosphere is found three times closer than the climate.");
  assert(abs(filter.x().at<sounded>() - truth.at<sounded>()) < delta<K>(1.) &&
         "The sounded temperature is found within a kelvin.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
