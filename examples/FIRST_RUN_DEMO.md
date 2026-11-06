# First-Run Demo and User Journey

This describes the **desired completed application experience**, not a screenshot of a finished app.

## 1. No files and no network

Launch `weightlab start`. UI opens locally. Choose `Explore Demo` to generate/open deterministic tiny base and fine-tune synthetic fixtures. No download, account or token. Import list shows a small transformer-like layer tree and exact raw tensor keys.

## 2. Explore weights

Click `demo.layers.0.mlp.up_proj.weight` and see an 8x8 heatmap with a scale legend. Hover row 3, column 5; tooltip displays exact original scalar, original dtype, matrix coordinate and `exact` quality. Drag a 4x4 ROI and open numeric table. Change color scaling; underlying values do not change.

## 3. Compare checkpoints

Select `tiny_base.safetensors` and `tiny_ft.safetensors`. Compatibility appears as `Class V: base file identity verified (SHA-256); synthetic pair, ancestry known by construction`. The overview ranks `demo.layers.0.mlp.up_proj.weight` first and every other tensor at zero change. Delta panel highlights exactly 10 cells changed, all others delta zero. Swap A/B and signs reverse. UI states: "These are known synthetic parameter changes; they are not evidence of improved reasoning." Export JSON and PNG with legend and provenance.

## 4. Explore a LoRA update

Import `tiny_lora/`. Choose a target module. See A matrix, B matrix and *effective* `scale*B@A` delta; inspect a known 2x2 tile and compare independently expected result. Experimental scale 0.5 is explicitly distinguished from adapter's native scaling. Original file remains unmodified.

## 5. Simulate a larger scan

Start exact stats on moderately sized fixture. UI shows estimates, progress, cancel. Cancel job; UI stays responsive and no partial result is mislabeled exact. Reopen app; bookmarks survive, running jobs show interrupted after restart.

## 6. Explain science caveat

Click "What can weights tell us?" and see:

- A heatmap shows numerical parameters, not neuron meanings.
- Large change magnitude isn't evidence of math capability gain.
- Comparing independently trained models may be invalid without alignment.
- To support a causal claim, run a controlled intervention and evaluation (optional R3).

## 7. Completion criteria

A first-time user with no coding environment can reach an exact numerical result, compare known deltas, save a report, and understand major interpretive limitations in one offline session. User can distinguish model bytes from app metadata and choose to delete just the project.
