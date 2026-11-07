# Coordinator guide: running the Phase 0.5 practitioner evaluation

Doc 14, Phase 0.5, task 6: "Put the CLI in front of 5-10 practitioners; record which questions it did not answer." Gate G0.5 needs a **written feedback summary that orders Phase 1/2 UI work**.

This kit produces the inputs for that summary. It does not produce the summary, and nothing in it counts as feedback until real participants have answered. **The G0.5 practitioner item stays OPEN** until a summary built from at least five returned questionnaires has been written and reviewed.

## 1. Build and check the kit

```bash
uv run python scripts/build_evaluation_kit.py          # -> dist/weightlab-evaluation-kit-<version>.zip
```

The archive contains:

- this directory;
- the wheel;
- the fixtures the examples use (plus `compat/`, precision and malformed fixtures for the optional cases);
- `weightlab-diff.md`;
- `SHA256SUMS`.

Before you send it, test it once on a clean machine or virtual environment, following `README.md` word for word.

## 2. Recruit 5–10 participants

- Aim for a mix of roles: at least two who fine-tune or ship adapters, and at least two researchers (interpretability, training dynamics or merging).
- Aim for a mix of operating systems: at least one Linux and one Windows user. Windows has not been validated yet (see `evidence/g0.5/G0.5_REPORT.md`).
- Prefer people who have their own checkpoints or adapters. Q11 is the strongest evidence.
- Do not recruit from the project team. Their answers do not count towards the five.

Assign IDs P01, P02, … in the order of invitation, and keep the ID-to-person mapping **outside the repository**.

## 3. Sessions

You can run sessions asynchronously (send the archive and `README.md`) or observed (30-minute call, participant shares their screen).

When observing:

- Do not explain the report before they answer Q4–Q10.
- Write down every question they ask aloud, word for word. Those are answers to Q5/Q8.
- Note where they hesitate.

## 4. Handling responses

- Store returned questionnaires as `P0n.md` in a private location.
- Commit only what the summary needs: aggregated counts, and verbatim quotes from participants who ticked Q15 consent.
- Remove confidential model names, tensor names and paths from quotes.
- Never edit an answer. Do not reword quotes. Do not fill gaps with guesses: an unanswered question is `no answer`.
- A response with failed `run_examples.py` checks is still a response. Record the failure in the summary's problems section.

## 5. Writing the summary

1. Copy `FEEDBACK_SUMMARY_TEMPLATE.md` to `evidence/g0.5/FEEDBACK_SUMMARY.md`.
2. Fill it in only from returned questionnaires and session notes.
3. Every statement in sections 4–7 cites participant IDs.
4. Have it reviewed by someone who did not write it.
5. The gate decision (section 8) is the reviewer's to make.
