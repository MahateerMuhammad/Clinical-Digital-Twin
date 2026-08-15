"""
scripts/dev/create_deterioration_notebook.py
─────────────────────────────────
Creates notebooks/11_deterioration_baseline.ipynb for the **landmark** Phase 5.

Why this generator was rewritten
────────────────────────────────
It previously emitted a notebook describing a "strict 6-hour prediction window"
and calling ``DeteriorationModelPipeline(window_hours=6.0)`` — the case-control
design superseded on 2026-08-06 by the landmark rebuild. Three claims in it were
false by the time it was read:

* the horizon, quoted as 6 hours where the promoted model predicts 48 hours from
  a 24-hour landmark — eight times more urgent, on patients already stable a day;
* the headline metrics, stated as "AUROC >0.99 and AUPRC >0.99" against the
  promoted model's 0.7679 / 0.1016;
* the explanation, attributing that performance to `vital_heart_rate_mean`,
  `vital_sbp_max` and NEWS2 composites — features the admission matrix does not
  contain, because `chartevents` is ICU-only and it carries no `vital_*` columns
  at all.

That was the sixth instance of one recurring defect in this project: a number
living somewhere that does not update when its source changes. The fix here is
structural rather than textual — this notebook no longer *states* the landmark,
horizon, winning model, feature count, or any metric. It **reads** the first four
from ``models/best_models/phase5_deterioration_landmark.json``, which `promote()`
writes, and **recomputes** the metrics by scoring the promoted model on the
held-out split. If Phase 5 is retrained at a different landmark, the notebook
follows automatically; it cannot go stale the way its predecessor did.

Regenerate with:
    python scripts/dev/create_deterioration_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/11_deterioration_baseline.ipynb
"""

# ── repo-root bootstrap ──────────────────────────────────────────────────────
# These scripts live two levels below the project root. Python puts the *script's*
# directory on sys.path, not the working directory, so `import src...` would fail
# from here; and many of them address data with root-relative paths such as
# "models/" or "reports/tables/". Both are fixed by putting the root on the path
# and running from it, which makes execution identical from any directory.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in text.strip("\n").split("\n")]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": [l + "\n" for l in text.strip("\n").split("\n")]}


def build_notebook() -> None:
    cells = [

        md("""
# Phase 5 — Clinical Deterioration as a Landmark Analysis

Predicting ward-to-ICU transfer for patients **still stable at a fixed landmark
time**, from laboratory trajectory, comorbidity and prior-utilisation features
observed strictly before that landmark.

> [!IMPORTANT]
> **Provenance.** This notebook supersedes the original 6-hour case-control design.
> Every parameter below is **read from the promoted model's metadata** and every
> metric is **recomputed from the served artifacts**, so this notebook cannot
> describe a model other than the one actually being served.
>
> The authoritative narrative is
> [`reports/phase5_clinical_deterioration_report.md`](../reports/phase5_clinical_deterioration_report.md).
> Unlike notebook 10, nothing here is typed in, so the two cannot disagree — but the
> report carries the clinical framing and the leakage audit that this notebook only
> asserts.
"""),

        md("""
## 1. The defect this design fixes

The original Phase 5 windowed features to `admittime + 24h` while the prediction
cutoff was nominally `t_event − 6h`. Those are different instants, and the transfer
time varies per patient. On this cohort **12,236 of 31,282 positive cases (39%)**
transferred to ICU before hour 24 — so their feature window reached *past the event*
and absorbed post-transfer ICU laboratory draws.

It showed plainly in the explanations: `lab_unique_items_24h` and
`lab_total_count_24h` ranked 1st and 3rd by SHAP. Both measure how much testing was
ordered, which is exactly what inflates once a patient reaches intensive care.

### The landmark framing

Fix a landmark **T**, and admit only patients still at risk at T — still in
hospital, not yet in ICU. Predict transfer in `(T, T + horizon]`.

This removes the leak *structurally* rather than by filtering columns: every patient
is event-free at the moment the feature window closes, so no feature can contain
post-event information. It also gives cases and controls an identical observation
window, which the case-control design did not.
"""),

        code("""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

sys.argv = ['notebook']  # the pipeline module parses argv at import time
from scripts.pipelines.run_deterioration_landmark import (
    build_landmark_cohort, design_matrix, is_volume_feature)

META = json.loads(
    (project_root / 'models/best_models/phase5_deterioration_landmark.json').read_text())

LANDMARK = META['landmark_hours']
HORIZON = META['horizon_hours']

display(Markdown(
    f"**Promoted model** — design `{META['design']}`, "
    f"landmark **{LANDMARK:.0f}h**, horizon **{HORIZON:.0f}h**, "
    f"winner **{META['winning_model']}** on {META['n_features']} features, "
    f"promoted {META['promoted_at'][:10]}."))
"""),

        md("""
## 2. Cohort at the landmark

"At risk" means both still admitted and still on the ward: a patient discharged
before T never faced the outcome, and a patient already in ICU at T has had it.
"""),

        code("""
cohort = build_landmark_cohort(LANDMARK, HORIZON)
y = cohort['deterioration'].to_numpy()
counts = cohort['split'].value_counts()

display(Markdown(
    "| | |\\n| :--- | ---: |\\n"
    f"| At-risk admissions at T={LANDMARK:.0f}h | {len(cohort):,} |\\n"
    f"| Deterioration events within {HORIZON:.0f}h | {int(y.sum()):,} |\\n"
    f"| Base rate | {y.mean()*100:.2f}% |\\n"
    f"| Train / val / test | {counts.get('train', 0):,} / "
    f"{counts.get('val', 0):,} / {counts.get('test', 0):,} |"))
"""),

        md("""
### 2a. The property that makes the window safe

This is the assertion the whole design exists to satisfy. Every patient in the
cohort is event-free at T by construction, so no feature observed up to T can carry
post-event information. If this ever fails, the rebuild has regressed to the defect
it replaced — so it is asserted here rather than described.
"""),

        code("""
# Nobody in the cohort had reached ICU at or before the landmark.
early = cohort.loc[cohort['t_icu'].notna() & (cohort['t_icu'] <= LANDMARK)]
assert early.empty, f'{len(early)} admissions were already in ICU at T'

# Every positive event falls strictly inside the horizon.
pos = cohort.loc[cohort['deterioration'] == 1, 't_icu']
assert (pos > LANDMARK).all() and (pos <= LANDMARK + HORIZON).all()

# Nobody was discharged before the landmark.
assert (cohort['t_dis'] > LANDMARK).all()

print(f'PASS  {len(cohort):,} admissions, all event-free at T={LANDMARK:.0f}h')
print(f'PASS  all {int(y.sum()):,} events fall in (T, T+{HORIZON:.0f}h]')
print('PASS  the feature window cannot reach past any event')
"""),

        md("""
## 3. Primary vs sensitivity feature sets

`DETERIORATION_EXCLUDE_STRICT` removes `*_count`, `*_abnormal_count` and
`*_missing_ratio` — but those globs do not match the `_24h`-suffixed variants, so
the windowed forms survived and dominated the original model.

Both sets are therefore built explicitly:

- **primary** — testing-volume and missingness features excluded; physiology only
- **sensitivity** — those features retained

The gap between them measures how much of the model reads *clinician concern*
rather than patient state. The primary set is the one promoted.
"""),

        code("""
X_primary = design_matrix(cohort, drop_volume=True)
X_sens = design_matrix(cohort, drop_volume=False)
volume = [c for c in X_sens.columns if is_volume_feature(c)]

print(f'primary     : {X_primary.shape[1]} features')
print(f'sensitivity : {X_sens.shape[1]} features')
print(f'testing-volume features held out of primary: {len(volume)}')
print('examples:', volume[:6])

assert X_primary.shape[1] == META['n_features'], (
    f"design matrix has {X_primary.shape[1]} features but the promoted model was "
    f"fitted on {META['n_features']}; the feature space has drifted")
print(f"\\nPASS  matches the promoted model's {META['n_features']} features")
"""),

        md("""
## 4. Scoring the promoted model

Metrics are computed here by loading the served artifacts and scoring them on the
held-out test split — not copied from the results table. The numbers below and those
in `reports/tables/deterioration_landmark_results.md` agree because they come from
the same model, not because they were kept in sync by hand.
"""),

        code("""
import joblib
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

model = joblib.load(project_root / 'models/best_models/phase5_deterioration_winning.pkl')
calib = joblib.load(project_root / 'models/best_models/phase5_deterioration_calibrated.pkl')

te = (cohort['split'] == 'test').to_numpy()
raw = model.predict_proba(X_primary[te])[:, 1]
cal = calib.predict(raw)

results = pd.DataFrame([
    {'Model': label,
     'AUROC': round(roc_auc_score(y[te], p), 4),
     'AUPRC': round(average_precision_score(y[te], p), 4),
     'Brier': round(brier_score_loss(y[te], p), 4)}
    for label, p in ((META['winning_model'], raw),
                     (f"{META['winning_model']} (Calibrated)", cal))
])
display(results)

base = y[te].mean()
print(f'test base rate   : {base*100:.2f}%')
print(f'AUPRC enrichment : {results.AUPRC.iloc[0] / base:.2f}x over chance')
"""),

        md("""
## 5. Calibration

At a ~2% base rate the raw scores are pushed toward the positive class by
`scale_pos_weight`, so they are not probabilities. Isotonic regression fitted on the
validation split maps them back onto the observed frequency scale — which is what
makes a number safe to put in front of a clinician.
"""),

        code("""
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for label, p in (('raw', raw), ('isotonic', cal)):
    frac, mean_pred = calibration_curve(y[te], p, n_bins=10, strategy='quantile')
    ax.plot(mean_pred, frac, marker='o', label=label)

lim = float(max(cal.max(), 0.2))
ax.plot([0, lim], [0, lim], 'k--', lw=1, label='perfect')
ax.set_xlabel('predicted probability')
ax.set_ylabel('observed frequency')
ax.set_title(f'Calibration — ICU transfer within {HORIZON:.0f}h of T={LANDMARK:.0f}h')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.legend()
plt.tight_layout(); plt.show()

print(f'Brier raw      : {brier_score_loss(y[te], raw):.4f}')
print(f'Brier isotonic : {brier_score_loss(y[te], cal):.4f}')
"""),

        md("""
## 6. What the model reads

SHAP on the promoted model. The check that matters here is a negative one: no
testing-volume or missingness feature should appear, because those are the features
that carried the original leak.
"""),

        code("""
import shap

sample = X_primary[te].head(2000)
shap_values = shap.TreeExplainer(model).shap_values(sample)

shap.summary_plot(shap_values, sample, max_display=15, show=False)
plt.title('Phase 5 (landmark) — SHAP feature importance')
plt.tight_layout(); plt.show()

order = np.argsort(np.abs(shap_values).mean(0))[::-1][:15]
top = [sample.columns[i] for i in order]
leaked = [c for c in top if is_volume_feature(c)]

print('top 15:')
for rank, name in enumerate(top, 1):
    print(f'  {rank:2d}. {name}')
assert not leaked, f'testing-volume features re-entered the model: {leaked}'
print('\\nPASS  no testing-volume feature in the top 15')
"""),

        md("""
## 7. Honest limitations

> [!WARNING]
> **Scope narrowed, deliberately.** Patients who deteriorate before the landmark are
> not represented. This model answers *"will a patient still stable at the landmark
> deteriorate over the horizon?"* — it does not cover early crashes, which need a
> separate model at a shorter landmark.

> [!WARNING]
> **The proxy is not the event.** MIMIC-IV has no code-blue or rapid-response table,
> so deterioration is proxied by ward-to-ICU transfer. That is a decision about beds
> as much as about physiology, and it inherits the admitting hospital's ICU capacity
> and triage culture.

> [!WARNING]
> **No vital signs.** `chartevents` is ICU-only in MIMIC-IV, so the admission matrix
> carries no `vital_*` columns and this ward model uses none — it reads laboratory
> trajectory, comorbidity and prior utilisation instead. Reaching the published band
> without physiology is notable, and it is also the clearest single opportunity to
> improve the model.

> [!NOTE]
> **Weaker than the number it replaces, and the number it replaces was not real.**
> Compare as enrichment rather than raw AUPRC — the base rate moved from 5.95% to
> ~2%, and AUPRC scales with base rate. See §5 of
> `reports/tables/deterioration_landmark_results.md`.
"""),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out = Path("notebooks/11_deterioration_baseline.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(cells)} cells)")
    print(f"execute with: jupyter nbconvert --to notebook --execute --inplace {out}")


if __name__ == "__main__":
    build_notebook()
