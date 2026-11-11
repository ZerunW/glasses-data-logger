import Darwin

/// Phone monotonic clock. Samples share this domain; glasses timestamps retain their own domain.
enum MonotonicClock {
  private static let timebase: mach_timebase_info_data_t = {
    var value = mach_timebase_info_data_t()
    mach_timebase_info(&value)
    return value
  }()

  static func nowNanoseconds() -> UInt64 {
    let ticks = mach_continuous_time()
    let numerator = UInt64(timebase.numer)
    let denominator = UInt64(timebase.denom)
    guard denominator != 0 else { return 0 }
    let (whole, wholeOverflow) = (ticks / denominator).multipliedReportingOverflow(by: numerator)
    let (fraction, fractionOverflow) = (ticks % denominator).multipliedReportingOverflow(by: numerator)
    if wholeOverflow || fractionOverflow { return UInt64.max }
    let (result, overflow) = whole.addingReportingOverflow(fraction / denominator)
    return overflow ? UInt64.max : result
  }
}
