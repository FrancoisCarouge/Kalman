::  _  __          _      __  __          _   _
:: | |/ /    /\   | |    |  \/  |   /\   | \ | |
:: | ' /    /  \  | |    | \  / |  /  \  |  \| |
:: |  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
:: | . \  / ____ \| |____| |  | |/ ____ \| |\  |
:: |_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

:: Kalman Filter for C++
:: Version 0.1.0
:: https://github.com/FrancoisCarouge/Kalman

:: SPDX-License-Identifier: Unlicense

:: This is free and unencumbered software released into the public domain.

:: Anyone is free to copy, modify, publish, use, compile, sell, or
:: distribute this software, either in source code form or as a compiled
:: binary, for any purpose, commercial or non-commercial, and by any
:: means.

:: In jurisdictions that recognize copyright laws, the author or authors
:: of this software dedicate any and all copyright interest in the
:: software to the public domain. We make this dedication for the benefit
:: of the public at large and to the detriment of our heirs and
:: successors. We intend this dedication to be an overt act of
:: relinquishment in perpetuity of all present and future rights to this
:: software under copyright law.

:: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
:: EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
:: MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
:: IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
:: OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
:: ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
:: OTHER DEALINGS IN THE SOFTWARE.

:: For more information, please refer to <https://unlicense.org>

@echo off
cd ..
cl ^
  /EHsc /O2 /MD /std:c++latest ^
  /D EIGEN_NO_MALLOC ^
  /D EIGEN_RUNTIME_NO_MALLOC ^
  /D NDEBUG ^
  /I include /I ^
    "F:\Drive\Projects\cpp\vcpkg\packages\eigen3_x86-windows\include\eigen3" /I ^
    "F:\Drive\Projects\cpp\vcpkg\packages\benchmark_x86-windows\include" ^
  "benchmark/*.cpp" ^
  /Fe:"kalman.exe" ^
  /link ^
    "F:\Drive\Projects\cpp\vcpkg\packages\benchmark_x86-windows\lib\benchmark.lib" ^
    "shlwapi.lib"
start "" /affinity 2 /Realtime "kalman.exe" --benchmark_filter="." --benchmark_out="benchmark/results.json"
cd benchmark
