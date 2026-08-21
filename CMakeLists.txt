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
#
# LANGUAGES NONE: C and CXX are enabled further down, after the CUDA
# host-compiler compatibility guard below has had a chance to run. That
# guard sometimes needs to select a specific CXX/C compiler *before* those
# languages are first enabled - once enable_language(CXX)/(C) has run,
# CMAKE_CXX_COMPILER/CMAKE_C_COMPILER are locked in for the rest of this
# configure run, so the guard would be too late if it ran afterward.
project(DeepCore
    VERSION 0.1.0
    DESCRIPTION "DeepCore multi-coin NVIDIA GPU miner"
    LANGUAGES NONE
)

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

# A ProgPowZ CUDA kernel (src/cuda/progpowz_kernel.cu) exists; its algorithm
# core is validated on CPU and (as of the last real-hardware run recorded in
# src/cuda/README.md) the GPU kernel itself has also been validated on real
# Volta hardware. This flag stays OFF by default so a plain CPU-only build
# never depends on a CUDA Toolkit being present. When turned ON without a
# working CUDA Toolkit installed, configuration degrades gracefully (a
# warning, CUDA targets skipped) rather than failing - see the
# check_language(CUDA) guard below. When a CUDA Toolkit *is* found but its
# nvcc does not support this machine's default C++ compiler as a host
# compiler, configuration fails with a clear, actionable error instead of
# silently building a mismatched binary that only fails at link time - see
# the "CUDA host-compiler compatibility guard" section below for why.
option(DEEPCORE_WITH_CUDA "Enable CUDA language/backend and build GPU kernel targets (requires the CUDA Toolkit)" OFF)

# -----------------------------------------------------------------------------
# CUDA host-compiler compatibility guard
#
# Background (the bug this guard exists to catch): nvcc only supports host
# C++ compilers up to some maximum version per CUDA Toolkit release. When
# CMake enables the CUDA language and the *system default* C++ compiler
# (e.g. `c++`/`g++`, whatever that resolves to) is too new for the detected
# CUDA Toolkit, CMake's own CUDA-compiler detection can silently fall back
# to a *different, older* host compiler for `.cu` translation units only -
# without that ever being surfaced to the rest of the project. The CUDA
# kernel then compiles fine and configuration succeeds with no warning at
# all. But every non-CUDA C++ translation unit in the same build (here:
# third_party/ethash-progpow-ref, linked into deepcore-gpu-selftest) is
# still compiled with the original, newer default compiler. Linking
# objects produced by two different major compiler versions into one
# binary is not guaranteed to be ABI-compatible, and when it isn't, the
# failure shows up only at the *final link* step, as an opaque undefined
# reference (e.g. `__cxa_call_terminate`) - by which point it looks nothing
# like a compiler-version problem.
#
# This guard runs *before* CXX/C are enabled (see the `LANGUAGES NONE`
# project() above) so that, when a mismatch is detected, it can either
# select a different CXX/C compiler for the *entire* project (not just the
# CUDA translation units - that is what actually prevents the ABI mismatch,
# since nvcc's own host-compiler choice alone cannot fix objects compiled
# elsewhere with a different compiler), or fail configuration outright with
# a clear explanation, rather than letting the mismatch surface later as a
# confusing link error.
# -----------------------------------------------------------------------------

set(DEEPCORE_CUDA_COMPILER_FOUND FALSE)

if(DEEPCORE_WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        set(DEEPCORE_CUDA_COMPILER_FOUND TRUE)

        # Returns TRUE/FALSE in ${out_var}: can nvcc actually use
        # ${compiler} as its -ccbin host compiler? Determined by literally
        # trying it on a trivial .cu file, rather than hard-coding a
        # CUDA-version-to-GCC-version compatibility table that would need
        # ongoing maintenance as new toolkit/compiler releases ship.
        function(deepcore_nvcc_accepts_host_compiler compiler out_var error_out_var)
            set(_probe_dir "${CMAKE_BINARY_DIR}/CMakeFiles/deepcore_cuda_hostcc_probe")
            file(MAKE_DIRECTORY "${_probe_dir}")
            set(_probe_src "${_probe_dir}/probe.cu")
            file(WRITE "${_probe_src}" "__global__ void deepcore_hostcc_probe_kernel() {}\n")
            execute_process(
                COMMAND "${CMAKE_CUDA_COMPILER}" -ccbin "${compiler}" -c "${_probe_src}"
                        -o "${_probe_dir}/probe.o"
                RESULT_VARIABLE _probe_rc
                OUTPUT_VARIABLE _probe_out
                ERROR_VARIABLE _probe_err
            )
            if(_probe_rc EQUAL 0)
                set(${out_var} TRUE PARENT_SCOPE)
                set(${error_out_var} "" PARENT_SCOPE)
            else()
                set(${out_var} FALSE PARENT_SCOPE)
                set(${error_out_var} "${_probe_out}\n${_probe_err}" PARENT_SCOPE)
            endif()
        endfunction()

        if(DEFINED CACHE{CMAKE_CXX_COMPILER})
            # User explicitly pinned the compiler (-DCMAKE_CXX_COMPILER=...).
            # Respect that choice rather than silently overriding it -
            # just validate it and fail clearly if it will not work.
            set(_deepcore_candidate "$CACHE{CMAKE_CXX_COMPILER}")
            deepcore_nvcc_accepts_host_compiler("${_deepcore_candidate}" _deepcore_ok _deepcore_probe_err)
            if(NOT _deepcore_ok)
                message(FATAL_ERROR
                    "DEEPCORE_WITH_CUDA is ON, but the detected CUDA Toolkit's "
                    "nvcc (${CMAKE_CUDA_COMPILER}) rejects the explicitly "
                    "pinned CMAKE_CXX_COMPILER (${_deepcore_candidate}) as a "
                    "host compiler - usually because it is newer than this "
                    "CUDA Toolkit release supports. Building anyway would not "
                    "fail cleanly: CMake can silently compile .cu files with a "
                    "different, older host compiler than the rest of the "
                    "project, producing ABI-mismatched objects that only fail "
                    "at link time (e.g. undefined reference to "
                    "__cxa_call_terminate).\n\n"
                    "nvcc's own error compiling with this host compiler:\n"
                    "${_deepcore_probe_err}\n\n"
                    "Fix: reconfigure with an older, CUDA-Toolkit-supported "
                    "host compiler pair, e.g.:\n"
                    "  cmake -S . -B build -DDEEPCORE_WITH_CUDA=ON "
                    "-DCMAKE_CXX_COMPILER=g++-<N> -DCMAKE_C_COMPILER=gcc-<N>\n"
                    "(install a compatible g++-<N>/gcc-<N> pair first if none "
                    "is present - check the CUDA Toolkit's release notes for "
                    "the maximum supported host compiler version).")
            endif()
        else()
            # No explicit pin: probe the default compiler CMake would
            # otherwise select (`c++`/`cc` on the PATH), and if that is not
            # accepted by nvcc, search installed g++-N/gcc-N pairs
            # (newest-compatible-first) and pre-seed CMAKE_CXX_COMPILER /
            # CMAKE_C_COMPILER with the first pair that works - for the
            # *whole* project, which is what actually avoids the
            # ABI-mismatch link failure this guard exists to prevent.
            find_program(_deepcore_default_cxx NAMES c++ g++)
            set(_deepcore_ok FALSE)
            if(_deepcore_default_cxx)
                deepcore_nvcc_accepts_host_compiler("${_deepcore_default_cxx}" _deepcore_ok _deepcore_probe_err)
            endif()

            if(_deepcore_ok)
                message(STATUS "DEEPCORE_WITH_CUDA: default CXX compiler "
                                "(${_deepcore_default_cxx}) is accepted by nvcc "
                                "(${CMAKE_CUDA_COMPILER}) as a host compiler - "
                                "using it for the whole project.")
            else()
                set(_deepcore_candidate_versions 16 15 14 13 12 11 10 9)
                set(_deepcore_found_version "")
                foreach(_v IN LISTS _deepcore_candidate_versions)
                    unset(_deepcore_gxx_candidate CACHE)
                    unset(_deepcore_gcc_candidate CACHE)
                    find_program(_deepcore_gxx_candidate g++-${_v})
                    find_program(_deepcore_gcc_candidate gcc-${_v})
                    if(_deepcore_gxx_candidate AND _deepcore_gcc_candidate)
                        deepcore_nvcc_accepts_host_compiler("${_deepcore_gxx_candidate}" _deepcore_v_ok _deepcore_v_err)
                        if(_deepcore_v_ok)
                            set(CMAKE_CXX_COMPILER "${_deepcore_gxx_candidate}" CACHE STRING "" FORCE)
                            set(CMAKE_C_COMPILER "${_deepcore_gcc_candidate}" CACHE STRING "" FORCE)
                            set(_deepcore_found_version "${_v}")
                            break()
                        endif()
                    endif()
                endforeach()

                if(_deepcore_found_version)
                    message(STATUS "DEEPCORE_WITH_CUDA: default CXX compiler "
                                    "(${_deepcore_default_cxx}) is too new for "
                                    "nvcc (${CMAKE_CUDA_COMPILER}); auto-selected "
                                    "g++-${_deepcore_found_version}/"
                                    "gcc-${_deepcore_found_version} instead, and "
                                    "will build the ENTIRE project with it (not "
                                    "just the CUDA translation units) to avoid "
                                    "ABI-mismatched objects at link time.")
                else()
                    message(FATAL_ERROR
                        "DEEPCORE_WITH_CUDA is ON and a CUDA Toolkit was found "
                        "(${CMAKE_CUDA_COMPILER}), but its nvcc rejects this "
                        "machine's default C++ compiler "
                        "(${_deepcore_default_cxx}) as a host compiler, and no "
                        "installed g++-<N>/gcc-<N> pair (tried versions "
                        "16 down to 9) was accepted as a replacement either.\n\n"
                        "nvcc's error compiling with the default compiler:\n"
                        "${_deepcore_probe_err}\n\n"
                        "Fix: install a g++/gcc version pair this CUDA Toolkit "
                        "supports (check its release notes for the maximum "
                        "supported host compiler version), e.g.:\n"
                        "  sudo apt-get install g++-13 gcc-13\n"
                        "then reconfigure (either let this guard auto-select it, "
                        "or pin it explicitly with -DCMAKE_CXX_COMPILER=g++-13 "
                        "-DCMAKE_C_COMPILER=gcc-13), or disable CUDA entirely "
                        "with -DDEEPCORE_WITH_CUDA=OFF.")
                endif()
            endif()
        endif()
    else()
        message(WARNING "DEEPCORE_WITH_CUDA is ON but no CUDA compiler was found - GPU kernel targets will be skipped.")
    endif()
endif()

# ----- Now that the compiler pair is finalized, enable C/CXX -----------------
enable_language(C)
enable_language(CXX)

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

# ----- Finish enabling CUDA, now that CXX/C (and therefore the host compiler
# CUDA must match) are settled ------------------------------------------------
if(DEEPCORE_WITH_CUDA AND DEEPCORE_CUDA_COMPILER_FOUND)
    # MUST be set BEFORE enable_language(CUDA) - CMake's enable_language(CUDA)
    # auto-populates CMAKE_CUDA_ARCHITECTURES itself (with nvcc's own
    # conservative fallback default, e.g. sm_52) as a side effect of enabling
    # the language, and caches it. If that happens first, the value is
    # already non-empty by the time a later `if(NOT CMAKE_CUDA_ARCHITECTURES)`
    # guard runs, so it silently never applies - real bug hit on a real GV100
    # build (Volta, sm_70): every kernel was actually compiled for sm_52, not
    # sm_70, without any error, because sm_52 PTX still runs (JIT-compiled)
    # on newer hardware; it's just the wrong architecture, and every
    # register/occupancy figure derived from that build's `ptxas -v` output
    # was for the wrong target.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        # This project's actual GPU fleet: Tesla V100-SXM2 (Volta,
        # sm_70) and Tesla A100 (Ampere, sm_80). Extend this list only
        # when a new architecture has real hardware to validate kernels
        # against - see the "refuse to start on an unsupported GPU"
        # requirement in the project's hardware-support policy.
        set(CMAKE_CUDA_ARCHITECTURES "70;80")
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Pin nvcc's host compiler to exactly match CMAKE_CXX_COMPILER (settled,
    # and already verified compatible, above). This is the crux of the whole
    # guard: without it, CMake may silently pick a *different* host compiler
    # for CUDA sources than CMAKE_CXX_COMPILER, which is exactly how the
    # link-time ABI mismatch this guard exists to prevent happened before.
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")

    # CUDA::nvrtc / CUDA::cuda_driver imported targets (for the per-period
    # NVRTC-compiled kernel - see src/cuda/progpowz_nvrtc_kernel.cpp)
    # aren't provided by enable_language(CUDA) alone; they need an
    # explicit find_package(CUDAToolkit) call.
    find_package(CUDAToolkit REQUIRED)

    # Embed progpowz_portable.hpp's content as a byte array for NVRTC to
    # use as a named header (NVRTC cannot read files from disk itself -
    # any header the runtime-compiled source #includes must be supplied
    # in memory - see progpowz_nvrtc_kernel.cpp). A byte array (via HEX
    # file reading), not a string literal, specifically to avoid any
    # quoting/escaping risk for arbitrary file content - validated
    # working (byte-exact round-trip) before being used here.
    #
    # NOTE: this runs at CONFIGURE time. If src/cuda/progpowz_portable.hpp
    # changes, re-run `cmake -S . -B build` (not just `cmake --build
    # build`) before trusting a build of NVRTC-related targets, or the
    # embedded copy will be stale.
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/progpowz_portable.hpp" _deepcore_ppz_portable_hex HEX)
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," _deepcore_ppz_portable_hexbytes "${_deepcore_ppz_portable_hex}")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
    file(WRITE "${CMAKE_BINARY_DIR}/generated/progpowz_portable_hpp_embedded.inc"
        "// Auto-generated by CMakeLists.txt from src/cuda/progpowz_portable.hpp at CONFIGURE time.\n"
        "// DO NOT EDIT BY HAND. Reconfigure (cmake -S . -B build) after editing\n"
        "// progpowz_portable.hpp to refresh this file - a plain `cmake --build` does not.\n"
        "static const unsigned char kEmbeddedProgpowzPortableHeaderBytes[] = {\n"
        "${_deepcore_ppz_portable_hexbytes}0x00\n"
        "};\n")
endif()

# Note: the deepcore-miner executable target itself (plus its warning flags,
# release optimization flags, and link options) is defined further down,
# after the optional network/mining libraries it conditionally links against
# - see the "Core target" section near the end of this file.

# -----------------------------------------------------------------------------
# Zano stratum wire protocol (pure encode/decode, no networking yet - see
# src/network/README.md). Uses nlohmann::json (single header, MIT license,
# https://github.com/nlohmann/json) for JSON parsing/building rather than a
# hand-rolled parser - this is exactly the "production JSON backend" the
# original stratum_client.hpp placeholder comment called out as still
# needed. Fetched at configure time and pinned to a specific release tag for
# reproducibility.
# -----------------------------------------------------------------------------
option(DEEPCORE_BUILD_NETWORK_PROTOCOL "Build the Zano stratum wire-protocol library and its self-test (fetches nlohmann::json)" ON)

if(DEEPCORE_BUILD_NETWORK_PROTOCOL)
    find_package(Threads REQUIRED)

    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
    )
    # nlohmann::json's own tests/docs/tools aren't needed - only the header.
    set(JSON_BuildTests OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(nlohmann_json)

    add_library(zano-stratum-protocol STATIC
        src/network/zano_stratum_protocol.cpp
    )
    target_include_directories(zano-stratum-protocol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(zano-stratum-protocol PUBLIC nlohmann_json::nlohmann_json)
    target_compile_options(zano-stratum-protocol PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )

    # CPU-only, no-network correctness check for the wire-protocol layer
    # (encode/decode against zanod's actual message shapes) - must pass
    # before any socket code is trusted to be talking the right protocol.
    add_executable(deepcore-protocol-selftest
        tools/protocol_selftest/protocol_selftest.cpp
    )
    target_link_libraries(deepcore-protocol-selftest PRIVATE zano-stratum-protocol)

    # TCP transport for the wire protocol above. Plain sockets (POSIX now;
    # Winsock code is written but has not been built/run on Windows - see
    # src/network/README.md) rather than a full async library, matching
    # this milestone's scope. Needs the stratum_client.hpp interface
    # target's include dir but not the interface's own compiled sources
    # (it's header-only).
    add_library(zano-stratum-client STATIC
        src/network/zano_stratum_client.cpp
    )
    target_include_directories(zano-stratum-client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(zano-stratum-client PUBLIC zano-stratum-protocol)
    target_link_libraries(zano-stratum-client PRIVATE
        $<$<PLATFORM_ID:Windows>:ws2_32>
    )
    target_compile_options(zano-stratum-client PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )
    set_target_properties(zano-stratum-client PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    # End-to-end transport test against a local mock server (see that
    # tool's header comment) - no real zanod daemon or network access
    # required or used.
    add_executable(deepcore-network-selftest
        tools/network_selftest/network_selftest.cpp
    )
    target_link_libraries(deepcore-network-selftest PRIVATE zano-stratum-client)
    target_link_libraries(deepcore-network-selftest PRIVATE
        $<$<PLATFORM_ID:Windows>:ws2_32>
    )
endif()

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

    # CPU-only correctness check for src/cuda/progpowz_codegen.hpp's
    # per-period program trace generator/interpreter (see that file's
    # header comment) - proves the trace-based (nonce-independent)
    # re-derivation of the mix program is exactly equivalent to
    # progpowz_hash_light()'s direct interpretation, across multiple
    # periods and nonces. Zero CUDA toolchain dependency. This must pass
    # before any CUDA source generated from the same trace can be trusted
    # to encode the right decisions (it does not by itself validate that
    # generated CUDA text - see that file's header comment).
    add_executable(deepcore-codegen-selftest
        tools/codegen_selftest/codegen_selftest.cpp
    )
    target_link_libraries(deepcore-codegen-selftest PRIVATE ethash-progpow-ref)
    target_include_directories(deepcore-codegen-selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)
    set_target_properties(deepcore-codegen-selftest PROPERTIES
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

        # Validation gate for the lane-cooperative (warp-shuffle) full-DAG
        # kernel - see tools/warp_kernel_selftest/warp_kernel_selftest.cu's
        # header comment. Not linked against deepcore-progpowz-work or
        # zano-stratum-protocol - it only needs the raw CUDA kernel
        # primitives (progpowz_kernel.cu, included directly) and the
        # vendored reference library, same as deepcore-gpu-selftest above.
        add_executable(deepcore-warp-kernel-selftest
            tools/warp_kernel_selftest/warp_kernel_selftest.cu
        )
        target_link_libraries(deepcore-warp-kernel-selftest PRIVATE ethash-progpow-ref)
        target_include_directories(deepcore-warp-kernel-selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)
        set_target_properties(deepcore-warp-kernel-selftest PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )

        # Validation gate for the per-period NVRTC-compiled kernel - see
        # tools/nvrtc_kernel_selftest/nvrtc_kernel_selftest.cu's header
        # comment. Mixed-language target: the .cu source (nvcc) includes
        # progpowz_kernel.cu directly for the already-validated comparison
        # baseline; progpowz_nvrtc_kernel.cpp (plain host C++, no CUDA
        # syntax - only Driver API/NVRTC header calls) is compiled by the
        # regular CXX compiler and linked in, same mixed-TU pattern
        # deepcore-miner itself already uses for progpowz_gpu_backend.cu.
        add_executable(deepcore-nvrtc-kernel-selftest
            tools/nvrtc_kernel_selftest/nvrtc_kernel_selftest.cu
            src/cuda/progpowz_nvrtc_kernel.cpp
        )
        target_link_libraries(deepcore-nvrtc-kernel-selftest PRIVATE ethash-progpow-ref CUDA::nvrtc CUDA::cuda_driver)
        target_include_directories(deepcore-nvrtc-kernel-selftest PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda
            ${CMAKE_BINARY_DIR}/generated
        )
        set_target_properties(deepcore-nvrtc-kernel-selftest PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
    endif()
endif()

# -----------------------------------------------------------------------------
# ProgPowZ mining-loop glue (job parsing, target comparison, share
# formatting) - pure data transformation between a pool job and the
# hashing core, no sockets/GPU here. Needs both the wire-protocol library
# (to parse a job's raw JSON payload) and the vendored reference library
# (its self-test cross-validates a Keccak-256 port against the real
# implementation), so it's only built when both are available.
# -----------------------------------------------------------------------------
if(DEEPCORE_BUILD_NETWORK_PROTOCOL AND DEEPCORE_BUILD_REFERENCE_TOOLS)
    add_library(deepcore-progpowz-work STATIC
        src/mining/progpowz_work.cpp
        src/mining/mining_loop.cpp
    )
    target_include_directories(deepcore-progpowz-work PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(deepcore-progpowz-work PUBLIC zano-stratum-protocol ethash-progpow-ref)
    target_link_libraries(deepcore-progpowz-work PRIVATE Threads::Threads)
    target_compile_options(deepcore-progpowz-work PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
    )

    add_executable(deepcore-mining-selftest
        tools/mining_selftest/mining_selftest.cpp
    )
    target_link_libraries(deepcore-mining-selftest PRIVATE deepcore-progpowz-work ethash-progpow-ref)

    # MiningLoop's architecture (job handling, cancellation, submission)
    # validated with a fake in-memory pool client - no real network needed.
    # See src/network/README.md for the separate real-daemon validation of
    # the pieces this loop wires together.
    add_executable(deepcore-mining-loop-selftest
        tools/mining_loop_selftest/mining_loop_selftest.cpp
    )
    target_link_libraries(deepcore-mining-loop-selftest PRIVATE deepcore-progpowz-work ethash-progpow-ref Threads::Threads)

    # GpuHashSearchBackend (src/cuda/progpowz_gpu_backend.{hpp,cu}) - the
    # mining_loop-facing GPU backend, wrapping the already real-hardware-
    # validated progpowz_light_kernel. Same gating as deepcore-gpu-selftest
    # (only built when a CUDA compiler was actually found) plus needs
    # deepcore-progpowz-work, hence nested here rather than in the
    # DEEPCORE_BUILD_REFERENCE_TOOLS block above.
    if(DEEPCORE_WITH_CUDA AND CMAKE_CUDA_COMPILER)
        add_executable(deepcore-gpu-backend-selftest
            tools/gpu_backend_selftest/gpu_backend_selftest.cu
            src/cuda/progpowz_gpu_backend.cu
        )
        target_link_libraries(deepcore-gpu-backend-selftest PRIVATE deepcore-progpowz-work ethash-progpow-ref)
        target_include_directories(deepcore-gpu-backend-selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)
        set_target_properties(deepcore-gpu-backend-selftest PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
        )
    endif()
endif()

# -----------------------------------------------------------------------------
# Core target - deepcore-miner
#
# Defined here (not near the top of this file) because its source selection
# depends on whether the optional network/mining layers above were built:
#   - When both DEEPCORE_BUILD_NETWORK_PROTOCOL and DEEPCORE_BUILD_REFERENCE_TOOLS
#     are ON (the default), this is the real CLI (src/cli/deepcore_miner_main.cpp)
#     wired to ZanoStratumClient + MiningLoop - an actually runnable miner.
#   - Otherwise, it falls back to the Phase 2 placeholder (src/main.cpp) with
#     no extra dependencies, so the project always configures and builds
#     regardless of which optional layers are enabled.
#   - When CUDA is additionally available (DEEPCORE_WITH_CUDA and a CUDA
#     compiler actually found), GpuHashSearchBackend is compiled in and the
#     CLI's --gpu flag becomes available; DEEPCORE_HAVE_GPU_BACKEND is
#     defined so the CLI source knows to expose it. Without CUDA, the CLI
#     still builds - it just doesn't offer --gpu, matching this project's
#     policy of never exposing a flag for something a given build can't do.
# -----------------------------------------------------------------------------
if(DEEPCORE_BUILD_NETWORK_PROTOCOL AND DEEPCORE_BUILD_REFERENCE_TOOLS)
    add_executable(deepcore-miner
        src/cli/deepcore_miner_main.cpp
    )
    target_link_libraries(deepcore-miner PRIVATE
        zano-stratum-client
        deepcore-progpowz-work
        Threads::Threads
    )

    if(DEEPCORE_WITH_CUDA AND CMAKE_CUDA_COMPILER)
        target_sources(deepcore-miner PRIVATE src/cuda/progpowz_gpu_backend.cu)
        target_compile_definitions(deepcore-miner PRIVATE DEEPCORE_HAVE_GPU_BACKEND)
    endif()
else()
    add_executable(deepcore-miner
        src/main.cpp
    )
endif()

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
#
# Scoped to $<COMPILE_LANGUAGE:CXX> because deepcore-miner may also contain
# a CUDA source (src/cuda/progpowz_gpu_backend.cu, added above when CUDA is
# available) - these are plain GCC/Clang/MSVC flags that nvcc's own
# front end does not reliably accept unscoped, unlike the CUDA-specific
# targets elsewhere in this file which never mix languages in one target.
# -----------------------------------------------------------------------------
target_compile_options(deepcore-miner PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-Wall -Wextra -Wpedantic>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
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
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>,$<BOOL:${DEEPCORE_ENABLE_NATIVE_OPT}>>:-ffast-math -march=native>
)

# MSVC: standard Release optimizations (/O2 maximize speed, /Oi intrinsics,
# /GL whole-program optimization) plus matching /LTCG at link time. No GCC
# flags are ever passed here.
target_compile_options(deepcore-miner PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2 /Oi /GL>
)
target_link_options(deepcore-miner PRIVATE
    $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/LTCG>
)

# -----------------------------------------------------------------------------
# Future extension points (documented, not yet wired up):
#   - enable_language(CUDA) + per-architecture object libraries under
#     src/cuda/sm_60, sm_70, sm_75, sm_80_86, sm_89, sm_90, sm_100_120,
#     selected via CMAKE_CUDA_ARCHITECTURES per build, once kernels exist.
#   - A GPU-backed IHashSearchBackend (see src/mining/README.md) wired into
#     deepcore-miner in place of/alongside CpuHashSearchBackend, once a
#     production kernel exists.
#   - install()/CPack rules for Linux tarball + checksum + SBOM generation,
#     and a separate hiveos/ packaging step, once there is a real binary to
#     package.
# -----------------------------------------------------------------------------

install(TARGETS deepcore-miner RUNTIME DESTINATION bin)
