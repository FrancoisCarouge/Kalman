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

#ifndef FCAROUGE_KALMAN_FILTER_INTERNAL_FACTORY_HPP
#define FCAROUGE_KALMAN_FILTER_INTERNAL_FACTORY_HPP

#include "type.hpp"
#include "x_z_p_q_r.hpp"
#include "x_z_p_q_r_h_f.hpp"
#include "x_z_p_q_r_hh_f_us_ps.hpp"
#include "x_z_p_q_r_hh_ff_ps.hpp"
#include "x_z_p_qq_rr_f.hpp"
#include "x_z_p_r.hpp"
#include "x_z_p_r_f.hpp"
#include "x_z_u_p_q_r.hpp"
#include "x_z_u_p_q_r_h_f_g_us_ps.hpp"
#include "x_z_u_p_qq_r_ff_gg_ps.hpp"

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace fcarouge::kalman_filter::internal {
// Converts the value of a declared characteristic to the filter member type.
template <typename To, typename From>
[[nodiscard]] constexpr auto convert(const From &value) -> To {
  return To(value);
}

template <typename To, typename Element, std::size_t Rows, std::size_t Columns>
[[nodiscard]] constexpr auto
convert(const row_list<Element, Rows, Columns> &value) -> To {
  return value.template to<To>();
}

// The filter deducer helps in selecting the filter type from the parameters
// declared by the caller. The filter deducer also helps in passing through or
// ignoring values for the filter construction. Finally the deducer helps in
// converting the parameters to the filter members types.
template <typename Filter = void> struct filter_deducer {
  template <typename... Arguments>
  [[nodiscard]] static constexpr auto
  operator()([[maybe_unused]] Arguments... arguments)
    requires(std::same_as<Filter, void>)
  {
    static_assert(false,
                  "This requested filter configuration is not yet supported. "
                  "Please, submit a pull request or feature request.");

    return x_z_p_r<double>{};
  }

  [[nodiscard]] static constexpr auto operator()() -> x_z_p_r<double>
    requires(std::same_as<Filter, void>);

  [[nodiscard]] static constexpr auto operator()() { return Filter{}; }

  template <typename X>
  [[nodiscard]] static constexpr auto operator()(state<X> x) {
    return x_z_p_r<X>(x.value);
  }

  template <typename X>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<X> z) {
    return x_z_p_r<X>(x.value);
  }

  template <typename X, typename Z>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z) {
    return x_z_p_q_r_h_f<X, Z>(x.value);
  }

  template <typename X, typename Z, typename U>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             [[maybe_unused]] input_t<U> u) {
    using kt = x_z_u_p_q_r_h_f_g_us_ps<X, Z, U, std::tuple<>, std::tuple<>>;

    return kt{typename kt::state(x.value)};
  }

  template <typename X, typename Z, typename U, typename P, typename Q,
            typename R, typename F, typename G, typename... Ps>
    requires requires() { requires std::invocable<Q, X, Ps...>; }
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             [[maybe_unused]] input_t<U> u, estimate_uncertainty<P> p,
             process_uncertainty<Q> q, output_uncertainty<R> r,
             state_transition<F> f, input_control<G> g,
             [[maybe_unused]] prediction_types_t<Ps...> pts) {
    using kt =
        x_z_u_p_qq_r_ff_gg_ps<X, Z, U, std::tuple<>,
                              repack<prediction_types_t<Ps...>>, Q, F, G>;

    return kt{typename kt::state(x.value),
              internal::convert<typename kt::estimate_uncertainty>(p.value),
              typename kt::noise_process_function(q.value),
              internal::convert<typename kt::output_uncertainty>(r.value),
              typename kt::transition_state_function(f.value),
              typename kt::transition_control_function(g.value)};
  }

  template <typename X, typename Z, typename U, typename... Us, typename... Ps>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             [[maybe_unused]] input_t<U> u,
             [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) {
    using kt = x_z_u_p_q_r_h_f_g_us_ps<X, Z, U, repack<update_types_t<Us...>>,
                                       repack<prediction_types_t<Ps...>>>;

    return kt{typename kt::state(x.value)};
  }

  template <typename X, typename Z, typename P, typename Q, typename R,
            typename H, typename T, typename O, typename... Us, typename... Ps>
    requires requires() {
      requires std::invocable<H, X, Us...>;
      requires std::invocable<T, X, Ps...>;
      requires std::invocable<O, X, Us...>;
    }
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             estimate_uncertainty<P> p, process_uncertainty<Q> q,
             output_uncertainty<R> r, output_model<H> h, transition<T> ff,
             observation<O> hh, [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) {
    using kt = x_z_p_q_r_hh_f_us_ps<X, Z, repack<update_types_t<Us...>>,
                                    repack<prediction_types_t<Ps...>>, H, T, O>;

    return kt{typename kt::state(x.value),
              internal::convert<typename kt::estimate_uncertainty>(p.value),
              internal::convert<typename kt::process_uncertainty>(q.value),
              internal::convert<typename kt::output_uncertainty>(r.value),
              typename kt::observation_state_function(h.value),
              typename kt::transition_function(ff.value),
              typename kt::observation_function(hh.value)};
  }

  template <typename X, typename Z, typename P, typename Q, typename R,
            typename H, typename F, typename O, typename... Ps>
    requires requires() {
      requires std::invocable<H, X>;
      requires std::invocable<F, Ps...>;
      requires std::invocable<O, X>;
    }
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             estimate_uncertainty<P> p, process_uncertainty<Q> q,
             output_uncertainty<R> r, output_model<H> hh,
             state_transition<F> ff, observation<O> obs,
             [[maybe_unused]] prediction_types_t<Ps...> pts) {
    using kt =
        x_z_p_q_r_hh_ff_ps<X, Z, repack<prediction_types_t<Ps...>>, H, F, O>;

    return kt{typename kt::state(x.value),
              internal::convert<typename kt::estimate_uncertainty>(p.value),
              internal::convert<typename kt::process_uncertainty>(q.value),
              internal::convert<typename kt::output_uncertainty>(r.value),
              typename kt::observation_state_function(hh.value),
              typename kt::transition_state_function(ff.value),
              typename kt::observation_function(obs.value)};
  }

  template <typename X, typename Z, typename P, typename Q, typename R,
            typename H, typename F>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             estimate_uncertainty<P> p, process_uncertainty<Q> q,
             output_uncertainty<R> r, output_model<H> h,
             state_transition<F> f) {
    using kt = x_z_p_q_r_h_f<X, Z>;

    return kt{typename kt::state(x.value),
              internal::convert<typename kt::estimate_uncertainty>(p.value),
              internal::convert<typename kt::process_uncertainty>(q.value),
              internal::convert<typename kt::output_uncertainty>(r.value),
              internal::convert<typename kt::output_model>(h.value),
              internal::convert<typename kt::state_transition>(f.value)};
  }

  template <typename X, typename Z, typename P, typename R>
    requires std::same_as<X, Z>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             estimate_uncertainty<P> p, output_uncertainty<R> r) {
    using kt = x_z_p_r<X>;

    return kt{typename kt::state(x.value),
              internal::convert<typename kt::estimate_uncertainty>(p.value),
              internal::convert<typename kt::output_uncertainty>(r.value)};
  }

  template <typename X, typename Z, typename P, typename R, typename F>
    requires std::same_as<X, Z>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             estimate_uncertainty<P> p, output_uncertainty<R> r,
             state_transition<F> f) {
    using kt = x_z_p_r_f<X>;

    return kt{typename kt::state(x.value),
              internal::convert<typename kt::estimate_uncertainty>(p.value),
              internal::convert<typename kt::output_uncertainty>(r.value),
              internal::convert<typename kt::state_transition>(f.value)};
  }

  template <typename X, typename Z, typename P, typename R>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             estimate_uncertainty<P> p, output_uncertainty<R> r) {
    using kt = x_z_p_q_r_h_f<X, Z>;

    return kt{.x = typename kt::state(x.value),
              .p =
                  internal::convert<typename kt::estimate_uncertainty>(p.value),
              .r = internal::convert<typename kt::output_uncertainty>(r.value)};
  }

  template <typename X, typename Z, typename P, typename R, typename F>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             estimate_uncertainty<P> p, output_uncertainty<R> r,
             state_transition<F> f) {
    using kt = x_z_p_q_r_h_f<X, Z>;

    return kt{.x = typename kt::state(x.value),
              .p =
                  internal::convert<typename kt::estimate_uncertainty>(p.value),
              .r = internal::convert<typename kt::output_uncertainty>(r.value),
              .f = internal::convert<typename kt::state_transition>(f.value)};
  }

  template <typename X, typename P, typename R, typename Q>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<X> z,
             estimate_uncertainty<P> p, process_uncertainty<Q> q,
             output_uncertainty<R> r) {
    using kt = x_z_p_q_r<X>;

    return kt{typename kt::state(x.value),
              internal::convert<typename kt::estimate_uncertainty>(p.value),
              internal::convert<typename kt::process_uncertainty>(q.value),
              internal::convert<typename kt::output_uncertainty>(r.value)};
  }

  template <typename X, typename Z, typename U, typename P, typename Q,
            typename R>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             [[maybe_unused]] input_t<U> u, estimate_uncertainty<P> p,
             process_uncertainty<Q> q, output_uncertainty<R> r) {
    using kt = x_z_u_p_q_r_h_f_g_us_ps<X, Z, U, std::tuple<>, std::tuple<>>;

    return kt{typename kt::state(x.value),
              internal::convert<typename kt::estimate_uncertainty>(p.value),
              internal::convert<typename kt::process_uncertainty>(q.value),
              internal::convert<typename kt::output_uncertainty>(r.value)};
  }

  template <typename X, typename Z, typename U, typename P, typename R,
            typename Q, typename... Us, typename... Ps>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             [[maybe_unused]] input_t<U> u, estimate_uncertainty<P> p,
             process_uncertainty<Q> q, output_uncertainty<R> r,
             [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) {
    using kt = x_z_u_p_q_r_h_f_g_us_ps<X, Z, U, repack<update_types_t<Us...>>,
                                       repack<prediction_types_t<Ps...>>>;

    return kt{typename kt::state(x.value),
              internal::convert<typename kt::estimate_uncertainty>(p.value),
              internal::convert<typename kt::process_uncertainty>(q.value),
              internal::convert<typename kt::output_uncertainty>(r.value)};
  }

  template <typename X, typename P, typename Q, typename R>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<X> z,
             [[maybe_unused]] input_t<X> u, estimate_uncertainty<P> p,
             process_uncertainty<Q> q, output_uncertainty<R> r) {
    using kt = x_z_u_p_q_r<X>;

    return kt{typename kt::state(x.value),
              internal::convert<typename kt::estimate_uncertainty>(p.value),
              internal::convert<typename kt::process_uncertainty>(q.value),
              internal::convert<typename kt::output_uncertainty>(r.value)};
  }

  template <typename X, typename Z, typename U, typename P, typename Q,
            typename R, typename H, typename F, typename G, typename... Us,
            typename... Ps>
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             [[maybe_unused]] input_t<U> u, estimate_uncertainty<P> p,
             process_uncertainty<Q> q, output_uncertainty<R> r,
             output_model<H> h, state_transition<F> f, input_control<G> g,
             [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) {
    using kt = x_z_u_p_q_r_h_f_g_us_ps<X, Z, U, repack<update_types_t<Us...>>,
                                       repack<prediction_types_t<Ps...>>>;

    return kt{typename kt::state(x.value),
              internal::convert<typename kt::estimate_uncertainty>(p.value),
              internal::convert<typename kt::process_uncertainty>(q.value),
              internal::convert<typename kt::output_uncertainty>(r.value),
              internal::convert<typename kt::output_model>(h.value),
              internal::convert<typename kt::state_transition>(f.value),
              internal::convert<typename kt::input_control>(g.value)};
  }

  template <typename X, typename Z, typename P, typename Q, typename R,
            typename F>
    requires requires() {
      requires std::invocable<Q, X>;
      requires std::invocable<R, X, Z>;
    }
  [[nodiscard]] static constexpr auto
  operator()(state<X> x, [[maybe_unused]] output_t<Z> z,
             estimate_uncertainty<P> p, process_uncertainty<Q> q,
             output_uncertainty<R> r, state_transition<F> f) {
    using kt = x_z_p_qq_rr_f<X, Z, Q, R>;

    return kt{typename kt::state(x.value),
              internal::convert<typename kt::estimate_uncertainty>(p.value),
              typename kt::noise_process_function(q.value),
              typename kt::noise_observation_function(r.value),
              internal::convert<typename kt::state_transition>(f.value)};
  }
};

template <typename Filter> inline constexpr filter_deducer<Filter> deducer{};

// The arithmetic type of a declaration, void otherwise.
template <typename Argument> struct arithmetic_of : std::type_identity<void> {};

template <typename Argument>
  requires arithmetic<typename Argument::type>
struct arithmetic_of<Argument> : std::type_identity<typename Argument::type> {};

template <typename Type, typename... Types>
concept agrees = ((std::same_as<Type, void> || std::same_as<Types, void> ||
                   std::same_as<Type, Types>) &&
                  ...);

template <typename... Types>
concept agree = (agrees<Types, Types...> && ...);

// Rejects mixed arithmetic types, which would narrow in the filter equations.
template <typename... Arguments> struct filter_deduction {
  static_assert(agree<typename arithmetic_of<Arguments>::type...>,
                "Declare the filter with a single arithmetic type.");

  using type = std::invoke_result_t<filter_deducer<>, Arguments...>;
};

template <typename... Arguments>
using deduce_filter = filter_deduction<Arguments...>::type;
} // namespace fcarouge::kalman_filter::internal

#endif // FCAROUGE_KALMAN_FILTER_INTERNAL_FACTORY_HPP
