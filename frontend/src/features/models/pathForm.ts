import { isAbsolutePath } from "../../lib/format";

export const PATH_MAX = 4096; // contract LocalPath maxLength
export const DISPLAY_NAME_MAX = 120;

export interface PathFields {
  entry_path: string;
  selected_root: string;
  display_name?: string;
}

export type FieldErrors = Partial<Record<keyof PathFields, string>>;

/** Client-side checks that mirror the contract; the server validates again (existence, containment, format). */
export function validatePathFields(f: PathFields): FieldErrors {
  const errors: FieldErrors = {};
  for (const k of ["entry_path", "selected_root"] as const) {
    const v = f[k];
    if (!v.trim()) errors[k] = "Required.";
    else if (v !== v.trim()) errors[k] = "Remove leading or trailing spaces.";
    else if (v.length > PATH_MAX) errors[k] = `At most ${PATH_MAX} characters.`;
    else if (v.includes("\0")) errors[k] = "Paths cannot contain NUL characters.";
    else if (!isAbsolutePath(v))
      errors[k] = "Use an absolute path, e.g. /Users/me/models/model.safetensors or C:\\models\\model.safetensors.";
  }
  if (f.display_name !== undefined && f.display_name.length > DISPLAY_NAME_MAX) {
    errors.display_name = `At most ${DISPLAY_NAME_MAX} characters.`;
  }
  return errors;
}
