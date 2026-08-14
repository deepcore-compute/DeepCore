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

# C is needed alongside C++ only for the vendored ProgPowZ reference code
# under third_party/ethash-progpow-ref, which has a handful of plain-C
# source files (Keccak permutation, prime-search helpers). deepcore-miner
# itself is pure C++.
project(DeepCore
    VERSION 0.1.0
    DESCRIPTION "DeepCore multi-coin NVIDIA GPU miner"
    LANGUAGES CXX C
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

# A ProgPowZ CUDA kernel (src/cuda/progpowz_kernel.cu) exists and its
# algorithm core has been validated on CPU (see
# tools/cuda_selftest/cuda_selftest.cpp / src/cuda/README.md), but the
# kernel itself has never been compiled or run on a GPU - this flag stays
# OFF by default until that has actually happened on real hardware. When
# turned ON without a working CUDA Toolkit installed, configuration
# degrades gracefully (a warning, CUDA targets skipped) rather than
# failing - see the check_language(CUDA) guard below.
option(DEEPCORE_WITH_CUDA "Enable CUDA language/backend and build GPU kernel targets (requires the CUDA Toolkit; kernel is unvalidated on real hardware so far)" OFF)

# Detect whether a CUDA compiler is actually available before committing to
# enable_language(CUDA), which would otherwise hard-fail configuration on
# any machine without the CUDA Toolkit (such as the one this project was
# initially scaffolded on). This lets DEEPCORE_WITH_CUDA be turned ON
# unconditionally in CI/build scripts without those failing on non-GPU
# build agents - it just skips the CUDA targets with a warning instead.
if(DEEPCORE_WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            # This project's actual GPU fleet: Tesla V100-SXM2 (Volta,
            # sm_70) and Tesla A100 (Ampere, sm_80). Extend this list only
            # when a new architecture has real hardware to validate kernels
            # against - see the "refuse to start on an unsupported GPU"
            # requirement in the project's hardware-support policy.
            set(CMAKE_CUDA_ARCHITECTURES "70;80")
        endif()
    else()
        message(WARNING "DEEPCORE_WITH_CUDA is ON but no CUDA compiler was found - GPU kernel targets will be skipped.")
    endif()
endif()

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
# ProgPowZ correctness reference (vendored, unmodified third-party code)
#
# This is Zano's own ProgPoW/Ethash reference implementation (see
# third_party/ethash-progpow-ref/README.md for provenance and license). It
# is built as a small, separate static library so the correctness-checking
# tool below can link it - it is never linked into deepcore-miner itself.
# -----------------------------------------------------------------------------
option(DEEPCORE_BUILD_REFERENCE_TOOLS "Build the vendored ProgPowZ reference and its test-vector generator" ON)

if(DEEPCORE_BUILD_REFERENCE_TOOLS)
    set(ETHASH_REF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ethash-progpow-ref)

    add_library(ethash-progpow-ref STATIC
        ${ETHASH_REF_DIR}/ethash.cpp
        ${ETHASH_REF_DIR}/managed.cpp
        ${ETHASH_REF_DIR}/progpow.cpp
        ${ETHASH_REF_DIR}/keccak.c
        ${ETHASH_REF_DIR}/keccakf800.c
        ${ETHASH_REF_DIR}/keccakf1600.c
        ${ETHASH_REF_DIR}/primes.c
    )

    # Vendored code as-is (see README.md: "not modified from upstream"), so
    # it is not held to this project's own warning/standard settings -
    # C++17 is what upstream targets and is fully valid to compile under a
    # C++20 toolchain without changes.
    target_include_directories(ethash-progpow-ref SYSTEM PUBLIC ${ETHASH_REF_DIR})
    set_target_properties(ethash-progpow-ref PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    add_executable(deepcore-gen-vectors
        tools/reference/gen_vectors.cpp
    )
    target_link_libraries(deepcore-gen-vectors PRIVATE ethash-progpow-ref)
    set_target_properties(deepcore-gen-vectors PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    # CPU-only correctness check for src/cuda/progpowz_portable.hpp (see
    # that file's header comment): compares our portable, host/device
    # -callable ProgPowZ port against the unmodified reference
    # implementation, with zero CUDA toolchain dependency. This must pass
    # before any CUDA kernel built on top of progpowz_portable.hpp can be
    # trusted.
    add_executable(deepcore-cuda-selftest
        tools/cuda_selftest/cuda_selftest.cpp
    )
    target_link_libraries(deepcore-cuda-selftest PRIVATE ethash-progpow-ref)
    target_include_directories(deepcore-cuda-selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)
    set_target_properties(deepcore-cuda-selftest PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )

    # Actual GPU correctness gate (see tools/cuda_selftest/gpu_selftest.cu
    # header comment) - only built when a CUDA compiler was actually found
    # above. This is the target that must be run on real Volta/Ampere
    # hardware before src/cuda/progpowz_kernel.cu can be trusted at all.
    if(DEEPCORE_WITH_CUDA AND CMAKE_CUDA_COMPILER)
        add_executable(deepcore-gpu-selftest
            tools/cuda_selftest/gpu_selftest.cu
        )
        target_link_libraries(deepcore-gpu-selftest PRIVATE ethash-progpow-ref)
        target_include_directories(deepcore-gpu-selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)
        set_target_properties(deepcore-gpu-selftest PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
    endif()
endif()

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
