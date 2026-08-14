cmake_minimum_required(VERSION 3.20)

# -----------------------------------------------------------------------------
# DeepCore - multi-coin NVIDIA GPU mining application
#
# This is the Phase 2 scaffolding build. Only header interfaces exist so far
# (src/hardware/gpu_manager.hpp, src/network/stratum_client.hpp); the CUDA
# kernels, coin-specific proof logic, and networking backend are intentionally
# not implemented yet. This file is written so that:
#   - it configures and builds cleanly with only the current stub sources
#   - future CUDA backends, JSON/networking libraries, and coin modules can be
#     added without restructuring the top-level layout
# -----------------------------------------------------------------------------

project(DeepCore
    VERSION 0.1.0
    DESCRIPTION "DeepCore multi-coin NVIDIA GPU miner"
    LANGUAGES CXX
)

# ----- Language standard ----------------------------------------------------
# C++20, no compiler-specific extensions (portable -std=c++20 / /std:c++20).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Emit compile_commands.json for tooling (clangd, IDE indexers). Harmless if unused.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to a Release build when the generator supports single-config builds
# (Makefiles/Ninja) and nothing was specified. Multi-config generators
# (Visual Studio) ignore CMAKE_BUILD_TYPE and use --config instead.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# Build options
# -----------------------------------------------------------------------------

# -march=native binds the binary to the build host's CPU instruction set
# (AVX2/AVX-512/etc.). This is fastest for a machine that builds and runs on
# itself (e.g. a HiveOS rig building locally), but it is NOT portable: a
# binary built with -march=native on one CPU can SIGILL on another. It must
# stay OFF for any binary that will be redistributed (releases, HiveOS
# packages) and reproducible-build checksums must be generated with it OFF.
option(DEEPCORE_ENABLE_NATIVE_OPT "GCC/Clang only: enable -march=native and -ffast-math (breaks portability & FP determinism, local-build-only)" OFF)

# CUDA kernels are not implemented yet. This flag is a placeholder for the
# milestone where enable_language(CUDA) and real .cu sources are added; it
# must not be turned on until sm_* backends exist and pass correctness tests.
option(DEEPCORE_WITH_CUDA "Enable CUDA language/backend (not yet implemented)" OFF)

# ----- Core target -----------------------------------------------------------
add_executable(deepcore-miner
    src/main.cpp
)

target_include_directories(deepcore-miner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(deepcore-miner PRIVATE
    DEEPCORE_VERSION_STRING="${PROJECT_VERSION}"
)

# GPU vendor SDKs (CUDA Toolkit, NVML) are deliberately NOT find_package()'d
# here. The GPU backend is designed around dynamic driver loading at runtime
# (dlopen()/dlsym() on Linux, LoadLibrary()/GetProcAddress() on Windows
# against libcuda.so / nvcuda.dll and libnvidia-ml.so / nvml.dll) so that the
# miner builds and runs on machines without the CUDA Toolkit installed, and
# so a HiveOS package does not need to bundle the Toolkit. When the dynamic
# loader implementation is added under src/hardware/, link the platform's
# dynamic-loading library here:
if(UNIX AND NOT APPLE)
    target_link_libraries(deepcore-miner PRIVATE ${CMAKE_DL_LIBS})
endif()
# On Windows, LoadLibrary/GetProcAddress live in kernel32, which is linked by
# default - no extra target_link_libraries entry is needed there.

# -----------------------------------------------------------------------------
# Compiler warnings (compiler-specific, via generator expressions)
# -----------------------------------------------------------------------------
target_compile_options(deepcore-miner PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# -----------------------------------------------------------------------------
# Release optimization flags
#
# GCC/Clang and MSVC accept different flag syntax, so each is scoped with a
# CXX_COMPILER_ID generator expression rather than set globally - a global
# CMAKE_CXX_FLAGS would otherwise pass GCC-style flags to MSVC and fail.
# -----------------------------------------------------------------------------

# GCC/Clang: -O3 always for Release. -ffast-math and -march=native are gated
# behind DEEPCORE_ENABLE_NATIVE_OPT since they are unsafe for redistributable
# binaries (see option() comment above). NOTE: -ffast-math relaxes IEEE-754
# semantics; it must be re-scoped OFF for any translation unit implementing
# consensus-critical hashing/proof code once that code exists, even when this
# option is enabled for the rest of the miner.
target_compile_options(deepcore-miner PRIVATE
    $<$<AND:$<CONFIG:Release>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:-O3>
    $<$<AND:$<CONFIG:Release>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>,$<BOOL:${DEEPCORE_ENABLE_NATIVE_OPT}>>:-ffast-math -march=native>
)

# MSVC: standard Release optimizations (/O2 maximize speed, /Oi intrinsics,
# /GL whole-program optimization) plus matching /LTCG at link time. No GCC
# flags are ever passed here.
target_compile_options(deepcore-miner PRIVATE
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2 /Oi /GL>
)
target_link_options(deepcore-miner PRIVATE
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/LTCG>
)

# -----------------------------------------------------------------------------
# Future extension points (documented, not yet wired up):
#   - enable_language(CUDA) + per-architecture object libraries under
#     src/cuda/sm_60, sm_70, sm_75, sm_80_86, sm_89, sm_90, sm_100_120,
#     selected via CMAKE_CUDA_ARCHITECTURES per build, once kernels exist.
#   - A JSON + async networking dependency (e.g. vcpkg/FetchContent-provided)
#     for src/network/stratum_client.cpp, kept out of this header-only phase.
#   - install()/CPack rules for Linux tarball + checksum + SBOM generation,
#     and a separate hiveos/ packaging step, once there is a real binary to
#     package.
# -----------------------------------------------------------------------------

install(TARGETS deepcore-miner RUNTIME DESTINATION bin)
