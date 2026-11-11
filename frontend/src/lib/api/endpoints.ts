// Typed wrappers for the implemented routes the UI uses (contracts/openapi.yaml 0.4.0).
// IDs are server-issued opaque strings; they are still encoded, and tensor names never enter a URL path.
import { request } from "./client";
import { toSearchParams, type TensorQuery } from "./tensorQuery";
import type { Health, Model, ModelImport, ModelRelink, Project, RoiRequest, RoiResponse, Settings, Tensor, TensorPage } from "./types";

const id = (v: string): string => encodeURIComponent(v);
type Sig = AbortSignal | undefined;

export const api = {
  health: (signal?: Sig) => request<Health>("/health", { auth: false, signal }),
  settings: (signal?: Sig) => request<Settings>("/settings", { signal }),

  listProjects: (signal?: Sig) => request<Project[]>("/projects", { signal }),
  createProject: (name: string) => request<Project>("/projects", { method: "POST", body: { name } }),
  getProject: (projectId: string, signal?: Sig) => request<Project>(`/projects/${id(projectId)}`, { signal }),
  deleteProject: (projectId: string) => request<undefined>(`/projects/${id(projectId)}`, { method: "DELETE" }),

  listModels: (projectId: string, signal?: Sig) => request<Model[]>(`/projects/${id(projectId)}/models`, { signal }),
  importModel: (projectId: string, body: ModelImport) =>
    request<Model>(`/projects/${id(projectId)}/models`, { method: "POST", body }),
  relinkModel: (modelId: string, body: ModelRelink) =>
    request<Model>(`/models/${id(modelId)}/relink`, { method: "POST", body }),

  listTensors: (modelId: string, q: TensorQuery, signal?: Sig) =>
    request<TensorPage>(`/models/${id(modelId)}/tensors`, { query: toSearchParams(q), signal }),
  getTensor: (modelId: string, tensorId: string, signal?: Sig) =>
    request<Tensor>(`/models/${id(modelId)}/tensors/${id(tensorId)}`, { signal }),

  /** Exact values of a bounded region (doc 09 section 2). The caller validates the answer with `parseRoi`. */
  roi: (body: RoiRequest, signal?: Sig) => request<RoiResponse>("/tensors/roi", { method: "POST", body, signal }),
};
