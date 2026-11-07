import { beforeEach, describe, expect, it } from "vitest";
import { clearToken, getToken, initToken, TOKEN_KEY } from "./token";

const T1 = "a".repeat(43);
const T2 = "B-_9".repeat(11);

function at(url: string) {
  window.history.replaceState(null, "", url);
}

describe("token bootstrap", () => {
  beforeEach(() => {
    clearToken();
    at("/");
  });

  it("moves the fragment token to sessionStorage and strips it from the address bar", () => {
    at(`/#token=${T1}`);
    expect(initToken()).toBe(T1);
    expect(sessionStorage.getItem(TOKEN_KEY)).toBe(T1);
    expect(window.location.hash).toBe("");
    expect(window.location.href).not.toContain(T1);
    expect(window.location.href).not.toContain("token");
  });

  it("never writes the token to localStorage or cookies", () => {
    at(`/#token=${T1}`);
    initToken();
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i) ?? "";
      expect(localStorage.getItem(k)).not.toContain(T1);
    }
    expect(document.cookie).not.toContain(T1);
  });

  it("keeps the query string and path when stripping", () => {
    at(`/?view=1#token=${T1}`);
    initToken();
    expect(window.location.pathname + window.location.search).toBe("/?view=1");
  });

  it("a new launch link rotates the stored token", () => {
    at(`/#token=${T1}`);
    initToken();
    at(`/#token=${T2}`);
    expect(initToken()).toBe(T2);
    expect(getToken()).toBe(T2);
    expect(sessionStorage.getItem(TOKEN_KEY)).toBe(T2);
  });

  it("reload without a fragment reuses the tab's token", () => {
    at(`/#token=${T1}`);
    initToken();
    at("/");
    expect(initToken()).toBe(T1);
  });

  it("rejects malformed tokens but still removes them from the URL", () => {
    at("/#token=short");
    expect(initToken()).toBeNull();
    expect(sessionStorage.getItem(TOKEN_KEY)).toBeNull();
    expect(window.location.hash).toBe("");
    at(`/#token=${"x".repeat(40)}<script>`);
    expect(initToken()).toBeNull();
  });

  it("returns null without any token and clearToken forgets it", () => {
    expect(initToken()).toBeNull();
    at(`/#token=${T1}`);
    initToken();
    clearToken();
    expect(getToken()).toBeNull();
  });
});
