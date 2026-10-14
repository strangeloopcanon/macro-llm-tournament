# Household response test: available evidence — 2026-10-14

We have real household purchase responses. The missing part is a matched set of
model predictions, not the absence of all household action data. The archive's
spending supplement was outside the active normalized survey table.

## Available data

| Material | Verified scope | Use |
| --- | --- | --- |
| Core SCE survey | 80,025 person-month rows; 10,749 respondents; January 2020–September 2025 | Demographics, survey weights, inflation/income/job expectations |
| SCE spending supplement | 32,897 rows; 14,710 respondents; 30 waves, December 2014–August 2024 | Expected purchases and later self-reported purchases |
| Matched four-month spending waves | 17,827 respondent transitions; 15,702 have all seven origin purchase probabilities | Candidate purchase-response evaluation population |
| Existing spending/belief bridge | 14,901 prepared rows across 14 waves | Previous development evidence; join and weighting need reconciliation |
| Fixed ecology cohort | 200 households; 1,400 history rows, March–September 2025 | Observed beliefs and explicit nonresponse; not observed spending actions |
| V24 model cache | 200 July responses, 800 retrospective responses, 200 feedback responses | Dollar policies and simulated choices; no matched purchase probabilities |
| V5 financial input | Same 200 households; corrected card-payment semantics | Prepared input only; no new model run |
| Macro JEPA | Restored checkpoint, datasets, readouts, predictions, and scores | Macro state machinery; not a trained household purchase model |

The spending supplement asks every four months. QSP4 gives each respondent's
probability of buying a large item during the next four months. QSP3 in the next
wave reports whether the respondent bought that item during the preceding four
months. This supports a direct expectation-to-action comparison.

Match QSP4 categories 1–6 to QSP3 categories 1–6. Match QSP4 category 7, a house,
to QSP3 category 9. QSP3 category 8 means no listed purchase; category 7 is other.
Check the questionnaire and glossary before implementing these codes. Missing
responses must remain missing. These are self-reported purchases, not bank records.
The first spending-wave pair lacks QSP4; usable expectation coverage starts later.

QSP1/QSP2 report spending growth over the previous year. Those windows overlap
between waves. They are a weaker target for the proposed four-month response test.
Hypothetical windfall allocations and expected spending are not realized actions.

## Existing test results and constraints

The earlier empirical bridge fitted spending responses from beliefs at group level.
It did not evaluate model predictions against individual purchase outcomes.
Its original chart reproduction gate failed. Later work relaxed chart tolerance and
used substitute core-survey weights. That is a limitation, not a clean validation.

The v5 bridge file says `status: accepted`, but its separate validation gate says
`passed: false`. Validation cell RMSE was 0.750 versus fit 0.473, a ratio of 1.585.
All three between-group sign/magnitude checks failed. Do not promote that bridge
from its status label. Validation includes December 2023, April 2024, and August
2024; those waves have already been inspected.

The spending workbook has no weight column. Same-wave core SCE weights are
available for some respondents. The saved bridge uses latest available core rows
and reports lags up to 92 days. A direct same-wave join has different coverage.
The saved bridge reports 6,822 respondents; the current same-wave join found 6,819.
Resolve this difference and publication timing before declaring a test population.

There are only 29 adjacent wave pairs, despite thousands of person rows. Shared
macro conditions vary by wave, and the same person can appear repeatedly. A model
comparison must account for both wave and respondent dependence. A random row split
would overstate the amount of independent evidence.

The cached ecology responses do not predict QSP4 category purchase probabilities.
Do not turn dollar spending policies into purchase probabilities after seeing labels.
No saved model-vs-purchase evaluation was found. Sealed survey waves remain unused.

## Recommended next step

Prepare an offline purchase-transition dataset from the restored supplement. Freeze
category mapping, missingness, eligible pairs, weights, and the split before fitting.
Use actual next-wave purchases as labels. Keep actions separate from beliefs.

First measure the respondents' own QSP4 forecasts, category base rates, and previous
purchases. Score probabilities with Brier score and log loss, plus calibration and
coverage. This establishes task difficulty without model calls or new training.

Then define one matched comparison: household history alone versus the same history
plus origin-safe JEPA macro state. Both must predict the same purchase categories
with the same eligible population and information cutoff. The current macro readout
cannot serve as a household prediction head without a separate fit and validation.

Use old waves as development evidence. Select a genuinely untouched later outcome
period for confirmation; do not relabel the old bridge validation as fresh. Resolve
whether the desired target is individual recurring customers or new respondents,
then make the person/time split match that target. Cluster uncertainty by wave and
person. Obtain exact publication dates for origin-safe public inputs.

No test dataset, baseline fit, model call, training run, or new paid experiment was
launched during this stocktake. The next useful authorization is a bounded offline
dataset and baseline study. A live household experiment should wait for its result.

## Recovered file locations

These raw paths now reside inside the verified Drive archives. The local `work/`
and `outputs/` folders retain restore pointers only. Reports and portable chart data
remain in the repository. Restore metadata accompanies each private archive.

- Core data: `work/persona_beliefs/sce_real_microdata.csv`.
- Cohort history: `work/persona_beliefs/persistent_household_scale_v23/selected_observed_history.csv`.
- Corrected input: `work/persona_beliefs/persistent_household_scale_v5/`.
- Spending raw data and questionnaire:
  `work/archive/20260714_superseded_by_household_ecology_v19/research/spending_survey_raw/`.
- Earlier bridge results:
  `work/archive/20260714_superseded_by_household_ecology_v19/research/empirical_bridge/`.
- Model checkpoint: `work/macro_jepa_v3/model/model.pt`.
- Macro release events: `work/fred_dense_events_event_jepa_v2/dense_macro_events.jsonl`.
- Historical nine-panel figure: `reports/current_macro_jepa_comparison.png`.
- Current scored August readout: [recovery and August score](recovery_and_august_score.md).
