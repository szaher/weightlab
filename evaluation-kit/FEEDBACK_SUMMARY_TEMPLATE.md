# G0.5 practitioner feedback summary

> **Template: contains no feedback.** Copy it to `evidence/g0.5/FEEDBACK_SUMMARY.md` and fill it in **only** from returned questionnaires (`QUESTIONNAIRE.md`) and observed-session notes.
>
> - Every claim in sections 4–7 cites participant IDs (e.g. `P02, P05`).
> - Unanswered means `no answer`, never a guess.
> - Quotes are verbatim and only from participants who gave consent in Q15.
> - Delete this box when you fill the template in.

- Kit version / commit:
- Collection period:
- Coordinator:
- Reviewer (did not write this summary):

## 1. Status

| Gate item (doc 14 Phase 0.5) | Status |
|---|---|
| ≥ 5 practitioners used the CLI | `_` of 5 required (`_` invited, `_` returned) |
| Questions the CLI did not answer are recorded | section 4: yes / no |
| Phase 1/2 UI work ordered from this feedback | section 7: yes / no |
| **Practitioner item of G0.5** | **OPEN** until the reviewer signs section 8 |

## 2. Participants

| ID | Role (Q1) | Sizes / adapters (Q2) | Current practice (Q3) | OS / CPU | Mode (async / observed) | Checks passed | Own model (Q11) |
|---|---|---|---|---|---|---|---|
| P01 | | | | | | | |

Coverage gaps (roles, OSs or model sizes not represented):

## 3. Ratings

Count the answers per value and give the median. Use `n/a` for answers marked not applicable.

| Question | 1 | 2 | 3 | 4 | 5 | n/a | Median | n |
|---|---|---|---|---|---|---|---|---|
| Q4 pair: what changed, where, how much | | | | | | | | |
| Q7 LoRA: effective update, rank, scale clear | | | | | | | | |
| Q9 incomplete report noticed unprompted | | | | | | | | |
| Q13 would use it | | | | | | | | |

Q9, where incompleteness was first noticed:

| Header | Exit status | Counts | Caveats | CSV/JSON | Not noticed |
|---|---|---|---|---|---|
| | | | | | |

Q10, preferred behaviour on unanalyzable parts:

| Refuse | Report part + mark incomplete | Approximate + label | Other |
|---|---|---|---|
| | | | |

## 4. Questions the CLI did not answer (gate item; Q5, Q8, Q11 and session notes)

Group the questions by theme. Keep one row per theme and list every participant who raised it.

| # | Theme (your grouping) | Example question, verbatim (consented) or paraphrased `[para]` | Raised by | n | Could Phase 1/2 answer it? (feature letter from Q12 or "new") |
|---|---|---|---|---|---|
| U1 | | | | | |

## 5. Confusing or misleading output (Q6, session notes)

| # | What (report element) | Participants' reading vs intended meaning | Raised by | Severity (misleads / slows / cosmetic) | Proposed fix |
|---|---|---|---|---|---|
| C1 | | | | | |

## 6. Blockers and problems (Q11, Q14)

| # | Kind (format / adapter type / size / time / install / bug) | Detail and error code | Raised by | Tracked as |
|---|---|---|---|---|
| B1 | | | | |

Own-model runs (Q11):

| ID | Size | Time | Matched expectations? | Notes |
|---|---|---|---|---|
| | | | | |

## 7. Feature priorities (Q12) and the proposed ordering of Phase 1/2 UI work

Scoring:

- Rank 1 = 3 points, rank 2 = 2 points, rank 3 = 1 point.
- Also report the number of participants who picked the feature at all.
- Ties are broken by section 4: the number of unanswered-question themes a feature would answer.

| Feature (Q12) | Points | Picked by (n) | Rank-1 by | Unanswered themes it answers (section 4) |
|---|---|---|---|---|
| a. Searchable tensor list + change stats | | | | |
| b. Single-matrix heatmap, exact hover | | | | |
| c. B − A difference heatmap | | | | |
| d. Exact ROI table + export | | | | |
| e. Per-tensor statistics | | | | |
| f. LoRA overlay | | | | |
| g. Layer/role dashboard | | | | |
| h. Multi-checkpoint timeline | | | | |
| i. Exports with provenance | | | | |
| j. Other (list) | | | | |

**Proposed order.** Each line cites the rows above that justify it. Note where it departs from the current doc 14 order (Phase 1 task 3 onwards); changing the order of the frozen roadmap needs an ADR.

1.
2.
3.

Minority views and disagreements worth keeping:

## 8. Decision

- [ ] At least 5 returned questionnaires from participants outside the project team
- [ ] Section 4 lists the unanswered questions, with participant IDs
- [ ] Section 7 orders the Phase 1/2 UI work and cites its evidence
- [ ] Confusing-output items (section 5) that mislead are filed as issues

Reviewer decision on the G0.5 practitioner item (passed / not passed, with reason):

Signed / date:

## 9. Raw data index

| ID | Returned (date) | Format | Stored at (private) | Quote consent | Follow-up consent |
|---|---|---|---|---|---|
| | | | | | |
