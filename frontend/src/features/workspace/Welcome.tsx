import type { Project } from "../../lib/api/types";
import { CreateProjectForm } from "../projects/CreateProjectForm";

/** First launch: no projects exist yet (doc 08 section 6). */
export function Welcome({ onCreated }: { onCreated: (p: Project) => void }) {
  return (
    <section className="welcome" aria-labelledby="welcome-title">
      <h1 id="welcome-title">Explore LLM weights on your computer.</h1>
      <p className="lede">
        Inspect SafeTensors checkpoints, understand tensor structure and investigate model changes. Your model files stay
        local. No GPU or cloud account is required.
      </p>
      <div className="card">
        <CreateProjectForm onCreated={onCreated} autoFocus />
      </div>
      <h2 className="h3">How importing works</h2>
      <ol className="steps">
        <li>Create a project. Projects hold the models you import and are saved on this computer.</li>
        <li>
          Paste the absolute path of a <code>.safetensors</code> file or a sharded model folder. WeightLab reads it in place:
          nothing is uploaded or copied.
        </li>
        <li>Search the tensor catalog and select a tensor to see its shape, dtype, size and location in the file.</li>
      </ol>
      <p className="help">
        Working from a clone of the WeightLab repository? <code>examples/fixtures/tiny_base.safetensors</code> is a small
        synthetic checkpoint you can import to try it out (use the <code>examples/fixtures</code> folder as the selected
        root). Installed packages do not include it.
      </p>
    </section>
  );
}
