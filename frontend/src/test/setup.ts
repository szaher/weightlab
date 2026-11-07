import "@testing-library/jest-dom/vitest";
import { cleanup, configure } from "@testing-library/react";
import { afterEach } from "vitest";

// Async queries wait up to 5 s: the ROI explorer debounces requests, and CI machines can be slow.
configure({ asyncUtilTimeout: 5000 });

// jsdom has no modal dialog implementation; model the open/close state the components rely on.
if (!HTMLDialogElement.prototype.showModal) {
  HTMLDialogElement.prototype.showModal = function showModal(this: HTMLDialogElement) {
    this.setAttribute("open", "");
  };
  HTMLDialogElement.prototype.close = function close(this: HTMLDialogElement) {
    this.removeAttribute("open");
  };
}

// jsdom has no 2D canvas; components draw only when a context exists (paint() is tested with a recording context).
HTMLCanvasElement.prototype.getContext = (() => null) as unknown as HTMLCanvasElement["getContext"];
// jsdom lacks PointerEvent; a MouseEvent carries the clientX/clientY the heatmap's hit test reads.
if (typeof window.PointerEvent === "undefined") {
  class PointerEventShim extends MouseEvent {}
  (window as unknown as { PointerEvent: typeof MouseEvent }).PointerEvent = PointerEventShim;
}

afterEach(() => {
  cleanup();
  sessionStorage.clear();
  localStorage.clear();
});
