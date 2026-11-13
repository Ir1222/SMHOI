# P7.0.2 — approved M1/M2 amendment record

Authorization: the user explicitly approved Actor B / Critic B+ and M1/M2 in
the P7.0.2 + P7.0.3 request, baseline `db4f71f0325d882bb528072cadd9db08e15946c0`.
The complete Master Plan and approved architecture review were read before edits.
These research-level changes were applied before source implementation.

| Master section changed | Amendment and reason |
|---|---|
| §1.3.4, P5/P7 rows | Preserve historical P5 interfaces; point prospective HOI policy work to §4.6. |
| §4.1 | Correct Object linear velocity wording to the existing absolute root-origin velocity in Robot heading axes; no velocity subtraction or code change. |
| New §4.6 | M1: freeze versioned Actor B and independent Critic B+ dimensions, order, frames, normalization, history/goals, migration and differentiability. |
| §10 P5 scope | M2: explicit prospective P7 HOI exception, including legacy P5 specification constraints. Historical P5 reports/results/tests remain unchanged. |
| §12 P7-A.2 | Distinguish unchanged physical WM architecture from the approved new policy architecture and named migration. |
| §12 P7-B.5 | Explicitly supersede Robot-only HOI Critic with B+1015; preserve original TD(lambda), masks and Robot-only defaults. |
| §12 P7-B acceptance | Check §4.6 HOI dimensions while preserving Robot-only dimensions; prohibit unapproved dimensions. |
| §14 boundaries | Limit the observation exception to §4.6; keep physical and contact-related exclusions. |
| §16 P7 instructions | Specify §4.6 and the software-only P7.0.2/0.3 stop point, preserving future P7-A/B approval gates. |
| §17.1 items 3 and 10 | Reconcile final acceptance with the separate HOI schema and unchanged physical/Robot-only contracts. |

M3, M4 and M5 were **not** applied. Measured force, actual contact, reference
contact intent, contact geometry observations, new prediction heads/losses,
PPO, reward changes, physical assets and training are not authorized here.
The architecture review itself remains the historical approval proposal.
No P5/P6 report, result or calibration is rewritten.
