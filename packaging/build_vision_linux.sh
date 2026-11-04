#!/usr/bin/env bash
# packaging/build_vision_linux.sh - the picture reader for a Linux engine package: strata-vision for
# x86_64 CUDA, on Basalt's own kernels (no ggml or llama.cpp).  The sibling of build_vision_mac.sh:
#
#   packaging/build_vision_linux.sh --basalt <the data/references/basalt checkout> --out <folder>
#
# <folder>/strata-vision is the program; install it beside `strata` as engine/strata-vision (the
# serve vision section's "exe") and commit the binary with the engine (git add -f: engine/ is
# ignored by directory).  Requires cmake, ninja, nvcc; sm_120 by default (STRATA_VISION_ARCH).
set -euo pipefail
basalt="" ; out="" ; arch="${STRATA_VISION_ARCH:-120}"
while [ $# -gt 0 ]; do
  case "$1" in
    --basalt) basalt="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$basalt" ] && [ -n "$out" ] || { echo "usage: $0 --basalt <basalt checkout> --out <folder>" >&2; exit 2; }
[ -x "$(command -v cmake)" ] || { echo "cmake not found" >&2; exit 2; }
[ -n "${CUDACXX:-}" ] || export CUDACXX="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT
# tools/vision is standalone except the `platform` host library the root project builds; the overlay
# provides exactly the root's platform sources (CMakeLists.txt:166) and includes the tool.
cat > "$build/CMakeLists.txt" <<CAMAKE
cmake_minimum_required(VERSION 3.24)
project(basalt_vision_linux CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES $arch)
set(VISION_ROOT $basalt)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
add_library(platform STATIC
  \${VISION_ROOT}/src/platform/file.cpp
  \${VISION_ROOT}/src/platform/memory.cpp
  \${VISION_ROOT}/src/platform/direct_file.cpp
  \${VISION_ROOT}/src/platform/process.cpp
  \${VISION_ROOT}/src/platform/mapped_file.cpp
  \${VISION_ROOT}/src/platform/cpu.cpp
  \${VISION_ROOT}/src/platform/system.cpp
  \${VISION_ROOT}/src/platform/dynlib.cpp)
target_include_directories(platform PUBLIC \${VISION_ROOT}/include)
target_link_libraries(platform PUBLIC \${CMAKE_DL_LIBS})
add_subdirectory(\${VISION_ROOT}/tools/vision \${CMAKE_BINARY_DIR}/vision EXCLUDE_FROM_ALL)
CAMAKE
cmake -S "$build" -B "$build/build" -G Ninja > /dev/null
cmake --build "$build/build" --target basalt-vision -j "$(nproc)" > /dev/null
mkdir -p "$out"
cp "$basalt/bin/basalt-vision" "$out/strata-vision"
echo "$out/strata-vision: $(du -h "$out/strata-vision" | cut -f1) - smoke test:"
printf 'ENC /nonexistent.png /dev/null\n' | timeout 60 "$out/strata-vision" --mmproj /dev/null 2>&1 | head -2 || true
