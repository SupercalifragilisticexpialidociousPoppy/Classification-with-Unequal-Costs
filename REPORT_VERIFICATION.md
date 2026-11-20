# Report verification plan

Status of the report as built from `main` at commit `ace417e` (21 Nov 2026), and the work left before submission, divided by member.
The compiled report is [`report/Project4_Report.pdf`](report/Project4_Report.pdf).

**How this works:** each task has an **owner** who fixes it and a **verifier** who checks it. The verifier is the reviewer the [team plan](Project4_Team_Work_Plan.pdf) already assigns to that owner. Tick a box only when its "done when" check passes. Line numbers refer to commit `ace417e`.

---

## 1. Where we stand

**Already verified (no action needed)**
- All 90 tests pass (`python -m pytest`, with `scikit-learn` installed).
- `python -m project4.run_all` runs every written experiment without errors, in about 40 s.
- Both classifiers, the calibration metrics and the prior-shift correction are mathematically correct and numerically stable.
- No experiment uses the test set to choose anything; hyperparameters and Platt scaling use validation only.
- C2 and C6 results reproduce when regenerated, up to floating-point noise. C6's synthetic results match the exact T6 curves (0.739 / 0.181 against 0.739 / 0.179).
- T1 (Section 2) and T6 (Section 7.1) are complete and correct.
- The report compiles to 26 pages with no LaTeX errors, once `run_all` has been run.

**Still to do:** the tasks in Section 2 below. In the current PDF these show up as:
- 3 broken references printed as **??**;
- 4 bold **[TODO: …]** notes in Section 4;
- 2 sections reading **"To be written by …"** (T2 in Section 3.1, C5 in Section 6.2);
- no C4 results in Section 5;
- a conclusion and contribution statement covering only Member 4.

## 2. Tasks by member

### Member 1: T1 + C5 · verifier: Member 5
- [ ] **M1-1 · Write the Task 4 experiment (blocking).** Create `project4/experiments/task4_prior_shift.py`; it's already listed in `run_all.py:18`.
  - Synthetic data: train at prior 0.5, deploy at 0.05, 0.1, 0.2, 0.4, 0.6 and 0.8.
  - Spambase: use `data.resample_to_prior` on the test part.
  - For both models, compare *uncorrected*, *corrected* (`prior_shift.adjust_posterior` or `bayes_action_shifted`) and *oracle* decisions.
  - Report realized loss against deployment prior (mean ± SE over 20 splits) and confusion matrices.
  - *Done when:* `run_all` shows `task4_prior_shift ok`, and its figure and tables are committed.
- [ ] **M1-2 · Write Section 6.2 (C5).** Replace "To be written by Member 1." (`report/sections/06_prior_shift.tex:147`) with the results of M1-1, loading its figure and tables from `project4/figures` and `project4/results`.
  - *Done when:* Section 6.2 shows the results, and the claims in the Section 6 checklist (lines ~137–138: "corrected decisions coincide with the oracle…") are backed by your numbers.
- [ ] **M1-3 · Fix the broken reference in the introduction.** `01_intro_setup.tex:41` cites `sec:task2`, but the label is `sec:task2-results`.
  - *Done when:* page 1 shows "Section 4" instead of "Section ??".
- [ ] **M1-4 · Add headline numbers to the introduction** (`01_intro_setup.tex:50`, TODO), once Sections 4–7 are final.

### Member 2: T2 + C3 · verifier: Member 3
- [ ] **M2-1 · Put T2 into the report (blocking).** The finished T2 theory is in `report/sections/03a_gaussian.tex`, but no file includes it. Section 3.1 therefore prints "To be written by Member 2." (`03_models.tex:12–15`).
  - Replace those placeholder lines with `\input{sections/03a_gaussian}`.
  - Remove the duplicate `\label{sec:t2}`, since 03a defines it too.
  - *Done when:* Section 3.1 shows the T2 derivation, and the "Section ??" references to `sec:t2-reg` (in 03a line 62 and `04_task2_results.tex:27`) resolve.
- [ ] **M2-2 · Commit the Task 2 outputs (blocking).** The report reads 8 files that aren't in the repo: `figures/task2_{loss,reject}_{spambase,synthetic}.png` and `results/task2_loss_*.tex`, `task2_confusion_*_logistic.tex`. Without them a fresh clone can't build the report.
  - *Done when:* after a fresh `git clone`, `cd report && pdflatex main` finds every file before `run_all` is run.
- [ ] **M2-3 · Fix the `run_all --only` crash.** `task2_cost_sweep.py:385–389`: `main(argv=None)` makes `argparse` read `run_all`'s own arguments. So `python -m project4.run_all --only task2_cost_sweep` exits with "unrecognized arguments: --only" and stops the whole run.
  - Use `def main(argv=())` and `main(sys.argv[1:])` under `if __name__ == "__main__":`.
  - *Done when:* `python -m project4.run_all --only task2_cost_sweep t1_expected_loss` runs both and prints the summary.
- [ ] **M2-4 · Write the 4 TODO discussions** in `04_task2_results.tex` (lines 115, 122, 130, 138), using the numbers in `results/task2_*.csv`.
  - *Done when:* no "[TODO" text remains in the PDF.
- [ ] **M2-5 · Explain or align the Task 2 costs.** Task 2 uses $c_I = 0.1$ and geometric costs (`task2_cost_sweep.py:74–75`): ratio 10 means $c_{FP} = 3.16$, $c_{FN} = 0.316$. Every other section uses 10 / 1 / 0.3 from `config.py`.
  - Either switch to `COST_SCHEME = "fn_fixed"` with `config.C_INSP`, or state clearly in Section 4 why the numbers differ from other sections.
- [ ] **M2-6 · Notation.** `03a_gaussian.tex` and `04_task2_results.tex` define their own `\cFP`, `\cFN`, `\cI` (upright) and use $\Sigma$ for covariances. `notation.tex` uses `\cfp`, `\cfn`, `\ci` and $S$. Switch to the shared macros.

### Member 3: T3 + C2 · verifier: Member 2
- [ ] **M3-1 · Make the scikit-learn tests skip, not fail, when scikit-learn is missing.** `tests/test_gaussian.py:68` and `:78` import `sklearn` directly; use `pytest.importorskip("sklearn")` like the other test files.
  - *Done when:* `python -m pytest` without scikit-learn shows skips and 0 failures.
- [ ] **M3-2 · Lead the preprocessing decision.** Your C2 experiment shows `log(1+x)` clearly improves the Gaussian fit (test log-loss 0.30 → 0.22), but only Task 3 uses it; Tasks 2 and 5 don't. As a result, logistic regression on Spambase scores 0.936 accuracy in Section 5 and 0.919 in Section 7.
  - Agree one setting with M2, M4 and M5. M6 puts it in `config.py` (M6-2).
- [ ] **M3-3 · Verify M2-1 to M2-6** as Member 2's verifier: check the T2 derivation now in Section 3.1 and the rerun Task 2 results.

### Member 4: T4 + C6 · verifier: Member 6
- [ ] **M4-1 · Update the contribution statement.** `A_contributions.tex:8` says your review of Member 5's C4 "is pending availability of that code". C4 is now merged, so do the review (M4-3) and update the sentence.
- [ ] **M4-2 · Collect the conclusion.** `08_conclusion.tex` has only your paragraph; the plan asks for one per member. Collect the other five (see "Everyone" below).
- [ ] **M4-3 · Verify Member 5's tasks (M5-1 to M5-4)** as their plan reviewer.

### Member 5: T5 + C4 · verifier: Member 4
- [ ] **M5-1 · Regenerate the Task 3 outputs with the team's models (blocking).** The committed `results/c4_*.tex` and `figures/c4_*` were made with scikit-learn stand-ins and print "PREVIEW: … regenerate after the team models are merged" into the report. Rerun `python -m project4.run_all --only task3_calibration` and commit.
  - *Done when:* no "PREVIEW" text remains in the outputs or the PDF.
- [ ] **M5-2 · Add the C4 results to Section 5.** Section 5 has only the T4 theory; your tables and figures (`c4_scores.tex`, `c4_temperature.tex`, `c4_reliability`, `c4_temperature`) never appear.
  - Add a C4 subsection with `\label{sec:c4}`. Section 6 already refers to it at `06_prior_shift.tex:121`.
  - *Done when:* Section 5 shows your results and the "Section ??" on page 16 resolves.
- [ ] **M5-3 · Register the T5 figure in `run_all`.** `experiments/t5_prior_correction.py` isn't in the `EXPERIMENTS` list in `run_all.py`, so its figure (used in Section 6) is never regenerated.
- [ ] **M5-4 · Fix the prior-shift test.** `tests/test_prior_shift.py:57` fails without scikit-learn (use `pytest.importorskip`), and uses `penalty="l2"`, which scikit-learn 1.8 deprecates. Drop the argument; L2 is the default.
- [ ] **M5-5 · Verify M1-1 and M1-2** as Member 1's verifier: the corrected decisions should match the oracle and beat the uncorrected ones.

### Member 6: C1 + T6 · verifier: Member 1
- [ ] **M6-1 · Verify Member 4's C6** (plan reviewer). Done: the inspection-band thresholds on log-odds are correct, shared code is used, the analytic overlay and $c^\ast$ are marked, SE over 20 seeds, no test-set tuning.
- [ ] **M6-2 · Centralise shared settings in `config.py`:** the agreed preprocessing (M3-2), the hyperparameter grids, and the Task 2 costs (M2-5). Make each experiment read them, so every section describes the same models.
- [ ] **M6-3 · Notation pass:** once M2-6 is done, check that every section uses the macros in `report/notation.tex`.
- [ ] **M6-4 · Final build:** once every box above is ticked, run the sign-off steps in Section 4 and recommit `report/Project4_Report.pdf`.

### Everyone
- [ ] **Conclusion:** one paragraph each in `report/sections/08_conclusion.tex` (send to M4, see M4-2).
- [ ] **Contribution statement:** one paragraph each in `report/sections/A_contributions.tex`.
- [ ] **Read your own sections in the PDF** and check that every number you quote matches the regenerated tables.

## 3. Section-by-section status

| Report section | Owner | Verifier | Status in `ace417e` |
|---|---|---|---|
| 1 Introduction and setup | M1, M6 | M5, M1 | broken reference `sec:task2`; headline numbers TODO |
| 2 Decision theory (T1) | M1 | M5 | ✅ complete |
| 3.1 Gaussian classifiers (T2) | M2 | M3 | ❌ placeholder; finished text not included (M2-1) |
| 3.2–3.3 Logistic regression, threshold cost (T3) | M3 | M2 | ✅ complete |
| 3.4 Gaussian classifiers on Spambase (C2) | M3 | M2 | ✅ complete |
| 4 Task 2 results (C3) | M2 | M3 | ⚠️ outputs not committed; 4 TODOs; own cost settings |
| 5 Scoring rules (T4) | M4 | M6 | ✅ complete |
| 5 Calibration results (C4) | M5 | M4 | ❌ missing; outputs stale (PREVIEW) |
| 6.1 Prior-shift correction (T5) | M5 | M4 | ✅ complete (figure not in `run_all`) |
| 6.2 Prior-shift experiments (C5) | M1 | M5 | ❌ not written; no experiment script |
| 7.1 Analytic model (T6) | M6 | M1 | ✅ complete |
| 7.2 Inspection experiments (C6) | M4 | M6 | ✅ complete, verified |
| 8 Conclusion | all | M4 | ⚠️ only Member 4 |
| Appendix: contributions | all | M4 | ⚠️ only Member 4; stale "pending" sentence |

## 4. Final sign-off (M6-4)

Run from the repository root on an up-to-date `main`:

```bash
python -m pip install -r requirements.txt
python -m pytest
python -m project4.run_all
cd report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The report is ready when all of these hold:
- [ ] `pytest`: all tests pass, with and without scikit-learn installed (skips allowed, failures not).
- [ ] `run_all`: every experiment shows `ok`, none `skipped` or `FAILED`.
- [ ] `git status` after `run_all` shows no new or changed result files (everything committed is up to date).
- [ ] `main.log` contains no `LaTeX Warning: Reference ... undefined` and no `Citation ... undefined`.
- [ ] The PDF contains no "??", "[TODO", "To be written", or "PREVIEW".
- [ ] Every requirement row in the plan's full-credit checklist has its evidence in the report.
- [ ] `report/Project4_Report.pdf` is recommitted from this build.
