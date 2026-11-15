import { describe, expect, it } from "vitest";
import { binInterval, edgeFractions, exactValue, magnitudeColumns, percent, shortValue, yFraction, yTicks } from "./chart";

describe("y scales", () => {
  it("linear is proportional and log is log10 over [1, max]", () => {
    expect(yFraction(50, 100, "linear")).toBe(0.5);
    expect(yFraction(0, 100, "linear")).toBe(0);
    expect(yFraction(100, 10_000, "log")).toBeCloseTo(0.5);
    expect(yFraction(0, 10_000, "log")).toBe(0);
    expect(yFraction(1, 10_000, "log")).toBeGreaterThan(0); // a single element stays visible
    expect(yFraction(1, 1, "log")).toBe(1);
  });

  it("ticks are round linear steps or powers of ten", () => {
    expect(yTicks(100, "linear")).toEqual([25, 50, 75, 100]);
    expect(yTicks(1234, "linear")).toEqual([500, 1000]);
    expect(yTicks(7, "linear")).toEqual([2, 4, 6]);
    expect(yTicks(12_345, "log")).toEqual([1, 10, 100, 1000, 10_000]);
    expect(yTicks(0, "linear")).toEqual([]);
  });
});

describe("edges and labels", () => {
  it("places bins at their true widths", () => {
    expect(edgeFractions([-1, 0, 2, 3])).toEqual([0, 0.25, 0.75, 1]);
    expect(edgeFractions([5, 5])).toEqual([0, 1]); // a constant tensor's single closed bin
  });

  it("states each interval with the backend's rule", () => {
    expect(binInterval([0, 0.5, 1], 0, "float", true)).toBe("[0, 0.5)");
    expect(binInterval([0, 0.5, 1], 1, "float", true)).toBe("[0.5, 1]");
    expect(binInterval([0, 0.5, 1], 1, "float", false)).toBe("[0.5, 1)");
    expect(binInterval([2, 2], 0, "float", true)).toBe("[2, 2]");
    expect(binInterval([-128, -102, 129], 0, "integer", false)).toBe("-128 … -103");
    expect(binInterval([3, 4], 0, "integer", false)).toBe("3");
  });

  it("exact values are the float64's shortest round trip; axis labels are short", () => {
    expect(exactValue(0.1 + 0.2)).toBe("0.30000000000000004");
    expect(exactValue(-0)).toBe("0");
    expect(exactValue("9007199254740993")).toBe("9007199254740993");
    expect(shortValue(0.30000000000000004)).toBe("0.3");
    expect(shortValue(1.2345e-7)).toBe("1.23e-7");
    expect(shortValue(-128)).toBe("-128");
  });

  it("fills empty exponents between the smallest and largest bucket", () => {
    expect(magnitudeColumns([{ exponent: -3, positive: 1, negative: 0 }, { exponent: -1, positive: 0, negative: 2 }])).toEqual([
      { exponent: -3, positive: 1, negative: 0 },
      { exponent: -2, positive: 0, negative: 0 },
      { exponent: -1, positive: 0, negative: 2 },
    ]);
    expect(magnitudeColumns([])).toEqual([]);
  });

  it("percent never rounds a nonzero share to 0", () => {
    expect(percent(1, 1_000_000)).toBe("<0.01%");
    expect(percent(0, 10)).toBe("0%");
    expect(percent(1, 3)).toBe("33.3%");
    expect(percent(1, 0)).toBe("—");
  });
});
