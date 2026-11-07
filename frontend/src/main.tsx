import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./app/App";
import { initToken } from "./lib/auth/token";
import "./styles/app.css";

initToken(); // before anything renders: take the #token fragment and strip it from the address bar

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}
