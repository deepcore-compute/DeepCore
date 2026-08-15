#pragma once

// deepcore::hardware - vendor-neutral GPU discovery, telemetry, and control
// interface.
//
// This header defines the abstract contract that a concrete NVIDIA backend
// (implemented separately, using dynamically-loaded NVML/CUDA driver
// entrypoints - see CMakeLists.txt) will implement. It intentionally
// contains no vendor SDK calls, no #include of any vendor header, and no
// hard-coded device count, indexing scheme, model name, VRAM size, or
// topology: all of that is discovered at runtime by enumerate_devices().

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace deepcore::hardware {

// -----------------------------------------------------------------------------
// Error model
// -----------------------------------------------------------------------------

// Error codes an IGpuManager implementation can report. Unsupported hardware
// controls (e.g. requesting a locked-clock override on a GPU/driver
// combination that does not expose it) MUST fail safely by returning
// UnsupportedControl (or a more specific code below) rather than silently
// doing nothing or attempting an unsafe fallback.
enum class GpuManagerErrc {
    Success = 0,
    DeviceNotFound,        // DeviceId does not correspond to a currently enumerated GPU
    DriverUnavailable,     // Vendor driver/library could not be loaded or initialized
    UnsupportedControl,    // The requested control is not exposed by this GPU/driver
    PermissionDenied,      // Control requires elevated privileges the process lacks
    InvalidArgument,       // Requested value is out of range (e.g. power limit)
    TelemetryUnavailable,  // A telemetry field could not be read (e.g. ECC on non-ECC card)
    Timeout,                // The underlying query/control call did not complete in time
};

[[nodiscard]] const std::error_category& gpu_manager_category() noexcept;

[[nodiscard]] inline std::error_code make_error_code(GpuManagerErrc e) noexcept
{
    return {static_cast<int>(e), gpu_manager_category()};
}

} // namespace deepcore::hardware

template <>
struct std::is_error_code_enum<deepcore::hardware::GpuManagerErrc> : std::true_type {};

namespace deepcore::hardware {

// -----------------------------------------------------------------------------
// Identity
// -----------------------------------------------------------------------------

// Stable handle used to address a specific physical GPU across calls. Index
// is the position reported by the current enumeration pass and is NOT
// guaranteed stable across driver reloads or GPU hot-plug; uuid is the
// vendor-reported stable identifier and should be preferred for logging,
// --devices filtering persistence, and cross-run comparisons.
struct DeviceId {
    std::uint32_t index{};
    std::string uuid;

    friend bool operator==(const DeviceId&, const DeviceId&) = default;
};

// CUDA compute capability, e.g. {8, 6} for sm_86 (Ampere).
struct ComputeCapability {
    int major{};
    int minor{};
};

enum class EccMode {
    Unknown,
    Unsupported,
    Disabled,
    Enabled,
};

// Static-ish identity/topology information for one GPU. Everything here is
// discovered at runtime by the concrete backend; nothing about GPU count,
// model, VRAM size, or bus layout may be assumed or hard-coded by callers.
struct GpuIdentity {
    DeviceId id;
    std::string model_name;          // e.g. "NVIDIA A100-SXM4-80GB"
    std::string pci_bus_id;          // e.g. "0000:65:00.0"
    ComputeCapability compute_capability;
    std::uint64_t vram_total_bytes{};
    std::optional<int> numa_node;    // absent if the platform/driver does not report one
    std::optional<unsigned> default_power_limit_watts;
    EccMode ecc_mode{EccMode::Unknown};
    std::string driver_version;
};

// -----------------------------------------------------------------------------
// Telemetry
// -----------------------------------------------------------------------------

// A point-in-time snapshot of one GPU's operating state. Any field the
// backend could not read on this call is left empty (std::nullopt) rather
// than defaulted to zero, so callers can distinguish "0" from "unknown".
struct GpuTelemetry {
    std::chrono::system_clock::time_point sampled_at;

    // Coin-agnostic proof/solution rate as measured by the mining loop, not
    // by this header - GpuManager only carries the value through for the
    // monitoring API. Units and meaning are defined by the caller (e.g.
    // proofs/sec); left empty when no work has completed yet.
    std::optional<double> hashrate;

    std::optional<unsigned> power_usage_watts;
    std::optional<unsigned> power_limit_watts;

    std::optional<int> temperature_celsius;
    std::optional<int> hotspot_temperature_celsius;
    std::optional<unsigned> fan_speed_percent;

    std::optional<unsigned> core_utilization_percent;
    std::optional<unsigned> memory_utilization_percent;
    std::optional<std::uint64_t> vram_used_bytes;

    std::optional<unsigned> core_clock_mhz;
    std::optional<unsigned> memory_clock_mhz;

    std::optional<std::uint64_t> ecc_single_bit_errors;
    std::optional<std::uint64_t> ecc_double_bit_errors;

    bool throttled{false};
};

// -----------------------------------------------------------------------------
// Control requests
// -----------------------------------------------------------------------------

// Absolute watts or a percentage of the GPU's default power limit; exactly
// one member should be set by the caller.
struct PowerLimitRequest {
    std::optional<unsigned> watts;
    std::optional<double> percent_of_default;
};

struct ClockLockRequest {
    std::optional<unsigned> core_clock_mhz;
    std::optional<unsigned> memory_clock_mhz;
};

struct FanControlRequest {
    // Empty means "return control to automatic/driver-managed fan curve".
    std::optional<unsigned> fan_percent;
};

// -----------------------------------------------------------------------------
// Manager interface
// -----------------------------------------------------------------------------

// Abstract, vendor-neutral entry point for GPU discovery, telemetry, and
// (where supported) hardware control.
//
// Thread-safety: implementations MUST be safe to call concurrently from
// multiple threads (e.g. a per-GPU mining thread and a separate monitoring/
// API thread reading telemetry at the same time). Callers may invoke any
// combination of methods on the same instance from different threads
// without external synchronization. Implementations are responsible for
// their own internal locking; this interface holds no mutable state itself.
//
// Ownership: callers do not own individual GPU handles - DeviceId is a
// lightweight value type, not a resource. The IGpuManager implementation
// owns the lifetime of any underlying driver/session handles and must
// release them in its destructor.
class IGpuManager {
public:
    virtual ~IGpuManager() = default;

    // Re-scans available devices. Implementations should call this once
    // internally before the first enumerate_devices()/read_telemetry() call;
    // callers may call it again to pick up hot-plug changes where the
    // platform supports it.
    virtual std::error_code refresh() = 0;

    // Returns identity/topology for every currently enumerated GPU. Order is
    // not guaranteed to match DeviceId::index across refresh() calls; use
    // DeviceId::uuid for persistent identification.
    [[nodiscard]] virtual std::vector<GpuIdentity> enumerate_devices() const = 0;

    // Reads a fresh telemetry snapshot for one device. Returns a non-success
    // error_code (DeviceNotFound, DriverUnavailable, ...) if the read could
    // not be performed at all; partial telemetry (some fields unknown) is
    // still reported as success with those fields left empty.
    [[nodiscard]] virtual std::error_code read_telemetry(const DeviceId& id, GpuTelemetry& out) const = 0;

    // Hardware control. All of these MUST fail safely (return
    // GpuManagerErrc::UnsupportedControl or a more specific code) and leave
    // the GPU's existing configuration untouched when the requested control
    // is not available on this GPU/driver/platform combination - they must
    // never silently no-op and report success, and never fall back to an
    // unrelated, potentially unsafe control.
    virtual std::error_code set_power_limit(const DeviceId& id, const PowerLimitRequest& request) = 0;
    virtual std::error_code set_locked_clocks(const DeviceId& id, const ClockLockRequest& request) = 0;
    virtual std::error_code set_fan_control(const DeviceId& id, const FanControlRequest& request) = 0;

    // Reverts any locked clocks/fan override applied by this process for the
    // given device back to driver-managed defaults. Used for graceful
    // shutdown and thermal-safety recovery. Must be safe to call even if no
    // override was ever applied (returns success as a no-op in that case).
    virtual std::error_code reset_controls(const DeviceId& id) = 0;
};

} // namespace deepcore::hardware
