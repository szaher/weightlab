// Non-secret, per-browser display preferences. The access token is never stored here.
export type Theme = "system" | "light" | "dark";
const THEME_KEY = "weightlab.pref.theme";

export function loadTheme(): Theme {
  try {
    const v = window.localStorage.getItem(THEME_KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

export function saveTheme(t: Theme): void {
  try {
    if (t === "system") window.localStorage.removeItem(THEME_KEY);
    else window.localStorage.setItem(THEME_KEY, t);
  } catch {
    /* preference not kept; harmless */
  }
}

export function applyTheme(t: Theme): void {
  if (t === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = t;
}
