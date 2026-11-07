// Human-readable explanation of an API failure (doc 08 section 5 "Error": copyable code, explanation and
// corrective action; no stack traces). Actions are only the ones that actually resolve the condition.
import { ApiRequestError } from "../api/client";

export interface ErrorView {
  title: string;
  explanation: string;
  action: string | null;
  code: string;
  requestId: string | null;
  kind: "auth" | "offline" | "error";
}

const RECONNECT =
  "Open the link that `weightlab start` printed in its terminal (it ends in #token=…). Restarting WeightLab prints a new link; old links stop working.";

/** Where the failure happened: import/relink forms, or an exact-value (ROI) read. */
export type ErrorContext = "import" | "relink" | "roi";

const ROI_FILES =
  "Values are never shown from a checkpoint that differs from the imported one. Re-import the model if it was modified; relink it in the Workspace if it was only moved.";

export function describeError(e: unknown, context?: ErrorContext): ErrorView {
  if (!(e instanceof ApiRequestError)) {
    return {
      title: "Something went wrong in the page",
      explanation: "The request could not be completed.",
      action: "Reload the page. If it keeps happening, restart `weightlab start`.",
      code: "CLIENT_ERROR",
      requestId: null,
      kind: "error",
    };
  }
  const base = { code: e.code, requestId: e.requestId ?? null, kind: "error" as const };
  const server = e.recommendedAction ?? null;
  const msg = e.message;
  switch (e.code) {
    case "NO_TOKEN":
    case "UNAUTHORIZED":
      return {
        ...base,
        kind: "auth",
        title: e.code === "NO_TOKEN" ? "Not connected to this WeightLab session" : "Access token expired or invalid",
        explanation:
          "Every WeightLab launch creates a new access token. This tab does not hold the token of the server that is running now.",
        action: RECONNECT,
      };
    case "NETWORK":
      return {
        ...base,
        kind: "offline",
        title: "WeightLab server unavailable",
        explanation: "The local server did not answer. It may have been stopped, or it is restarting.",
        action: "Start it again with `weightlab start` and open the new link it prints.",
      };
    case "FORBIDDEN_ORIGIN":
      return {
        ...base,
        title: "Request refused by the local server",
        explanation: "The server accepts requests only from its own page on 127.0.0.1, localhost or [::1].",
        action: "Open WeightLab from the exact link printed by `weightlab start`.",
      };
    case "RATE_LIMITED":
      return { ...base, title: "Too many requests at once", explanation: msg, action: server ?? "Retry in a moment." };
    case "RESOURCE_LIMIT":
      if (context === "roi") {
        return {
          ...base,
          title: "Region exceeds the exact-read budget",
          explanation: msg,
          action: "Zoom in to a smaller window. The budget is the server's roi_max_cells setting (1–65,536 cells).",
        };
      }
      return { ...base, title: "Request exceeds a resource limit", explanation: msg, action: server };
    case "INVALID_ROI":
      return {
        ...base,
        title: "Region or view not valid for this tensor",
        explanation: msg,
        action: "Reset to the origin, or choose axes and slice indices within the tensor's shape.",
      };
    case "UNSUPPORTED_PRECISION":
      return {
        ...base,
        title: "Values cannot be represented exactly",
        explanation: msg,
        action: "WeightLab does not show values it cannot represent exactly. This view is unavailable for this tensor.",
      };
    case "FILE_CHANGED":
      return {
        ...base,
        title: context === "relink" ? "These files are not the imported ones" : "Model files changed on disk",
        explanation: msg,
        action:
          context === "relink"
            ? "Relink accepts only the exact files that were imported. If the model was modified, import it again as a new model."
            : context === "roi"
              ? ROI_FILES
              : (server ?? "Re-import the model; relink only if it was moved unchanged."),
      };
    case "PATH_OUTSIDE_ROOT":
      return {
        ...base,
        title: "Path is outside the selected root",
        explanation: msg,
        action: server ?? "Choose a selected root that contains the model and all of its shards.",
      };
    case "INVALID_FORMAT":
    case "MISSING_SHARD":
    case "UNSUPPORTED_DTYPE":
      if (context === "roi") {
        return {
          ...base,
          title: e.code === "UNSUPPORTED_DTYPE" ? "This dtype cannot be read exactly" : "The model's files cannot be read",
          explanation: msg,
          action: e.code === "UNSUPPORTED_DTYPE" ? "Its metadata stays available above." : ROI_FILES,
        };
      }
      return {
        ...base,
        title: e.code === "MISSING_SHARD" ? "A shard of this model is missing" : "Not a supported SafeTensors checkpoint",
        explanation: msg,
        action:
          "Supported: a .safetensors file, a folder of sharded SafeTensors with model.safetensors.index.json, or that index file. Files such as .bin, .pt or .gguf are not read (no code from model files is ever run).",
      };
    case "NOT_FOUND":
      if (context === "import" || context === "relink") {
        return {
          ...base,
          title: "Path not found",
          explanation: msg,
          action: "Check the spelling of the absolute path. The path is read on the computer running WeightLab.",
        };
      }
      if (context === "roi") {
        return { ...base, title: "Tensor no longer available", explanation: msg, action: "The model may have been removed. Reload the model list." };
      }
      return { ...base, title: "Not found", explanation: msg, action: "It may have been deleted. Reload the list." };
    case "VALIDATION_ERROR":
      if (e.status === 403 || msg.startsWith("permission denied")) {
        return {
          ...base,
          title: "Permission denied",
          explanation: msg,
          action:
            "Your operating system does not let WeightLab read this location. Grant your user read access to the folder (or choose another copy); WeightLab never needs administrator rights.",
        };
      }
      return { ...base, title: "Invalid input", explanation: msg, action: server };
    case "BAD_RESPONSE":
      return { ...base, title: "Unexpected server response", explanation: msg, action: "Reload the page." };
    default:
      return { ...base, title: "Request failed", explanation: msg, action: server };
  }
}
