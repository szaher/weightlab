import { describe, expect, it } from "vitest";
import { DEFAULT_QUERY, pageInfo, parseCount, toggleSort, toSearchParams, updateQuery } from "./tensorQuery";

describe("tensor query serialization", () => {
  it("sends only sort and paging by default", () => {
    expect(toSearchParams(DEFAULT_QUERY).toString()).toBe("sort=raw_key&order=asc&limit=50&offset=0");
  });

  it("serializes every filter with the contract's parameter names", () => {
    const p = toSearchParams({
      ...DEFAULT_QUERY,
      q: "  layers.0 ",
      dtype: "BF16",
      role: "attention.q",
      layer: "3",
      rank: "2",
      minNumel: "1000",
      sort: "numel",
      order: "desc",
      limit: 100,
      offset: 200,
    });
    expect(p.get("q")).toBe("layers.0");
    expect(p.getAll("dtype")).toEqual(["BF16"]);
    expect(p.getAll("role")).toEqual(["attention.q"]);
    expect(p.get("layer")).toBe("3");
    expect(p.get("rank")).toBe("2");
    expect(p.get("min_numel")).toBe("1000");
    expect(p.get("sort")).toBe("numel");
    expect(p.get("order")).toBe("desc");
    expect(p.get("limit")).toBe("100");
    expect(p.get("offset")).toBe("200");
  });

  it("escapes special characters in the search text", () => {
    const p = toSearchParams({ ...DEFAULT_QUERY, q: "a&b=c#d/é" });
    expect(p.toString()).toContain("q=a%26b%3Dc%23d%2F%C3%A9");
    expect(new URLSearchParams(p.toString()).get("q")).toBe("a&b=c#d/é");
  });

  it("caps search length at the contract limit", () => {
    expect(toSearchParams({ ...DEFAULT_QUERY, q: "x".repeat(300) }).get("q")).toHaveLength(256);
  });

  it("drops invalid numeric filters instead of sending them", () => {
    const p = toSearchParams({ ...DEFAULT_QUERY, layer: "-1", rank: "33", minNumel: "1e3" });
    expect(p.has("layer")).toBe(false);
    expect(p.has("rank")).toBe(false);
    expect(p.has("min_numel")).toBe(false);
    expect(parseCount("12")).toBe(12);
    expect(parseCount(" 7 ")).toBe(7);
    expect(parseCount("1.5")).toBeNull();
    expect(parseCount("99999999999999999")).toBeNull();
  });
});

describe("paging and sorting", () => {
  it("resets to the first page when anything but the offset changes", () => {
    const q = { ...DEFAULT_QUERY, offset: 150 };
    expect(updateQuery(q, { dtype: "F32" }).offset).toBe(0);
    expect(updateQuery(q, { q: "x" }).offset).toBe(0);
    expect(updateQuery(q, { limit: 100 }).offset).toBe(0);
    expect(updateQuery(q, { offset: 200 }).offset).toBe(200);
  });

  it("toggles sort order on the same column and starts ascending on a new one", () => {
    const a = toggleSort({ ...DEFAULT_QUERY, offset: 100 }, "raw_key");
    expect([a.sort, a.order, a.offset]).toEqual(["raw_key", "desc", 0]);
    const b = toggleSort(a, "raw_key");
    expect(b.order).toBe("asc");
    const c = toggleSort(a, "numel");
    expect([c.sort, c.order]).toEqual(["numel", "asc"]);
  });

  it("computes page ranges", () => {
    expect(pageInfo(120, 0, 50, 50)).toMatchObject({ first: 1, last: 50, page: 1, pages: 3, hasPrev: false, hasNext: true, lastOffset: 100 });
    expect(pageInfo(120, 100, 50, 20)).toMatchObject({ first: 101, last: 120, page: 3, hasNext: false, hasPrev: true });
    expect(pageInfo(0, 0, 50, 0)).toMatchObject({ first: 0, last: 0, pages: 1, hasNext: false });
  });
});
