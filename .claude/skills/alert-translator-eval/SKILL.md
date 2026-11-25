---
name: alert-translator-eval
description: Change or evaluate how operator alerts become schedule overrides - the Claude prompt/schema/guardrails in ai_harness/llm_translator.py, the regex rules in ai_harness/exception_translator.py, or the labelled cases in ai_harness/evals/alert_cases.jsonl. Measures accuracy before and after, so a change is never judged by eye.
---

# Alert translator: change it, then measure it

Alerts flow: text → Claude extraction (`AlertExtraction`, structured output) → `extraction_to_patch`
(deterministic guardrails) → `ScheduleOverridePatch` → optimizer. The regex parser
(`translate_exception_text`) is the offline baseline and the fallback.

## Non-negotiable design rules

1. **Claude extracts, code decides.** Site policy (for example the 75% storm reserve) and every numeric
   limit live in `extraction_to_patch`, never in the prompt.
2. **Overrides only tighten.** `ScheduleOverridePatch.apply` takes max for floors and min for caps, and
   `run_scenario` never lowers the site's end-SoC floor. Keep it that way.
3. **Clamp everything.** Hours stay inside the horizon, 0 ≤ reserve ≤ capacity, caps ≥ 0. Record each
   clamp in `patch.notes`.
4. **Never drop a request silently.** Anything unmappable goes to `not_applied` and is shown to the operator.
5. **Fail safe.** `mode="auto"` must fall back to rules on any API failure, refusal or truncated output.

## Workflow

1. Record the baseline before you edit anything:
   ```bash
   python -m ai_harness.evals.run_alert_eval --mode rules
   ```
2. Make the change. If you change behaviour, add eval cases first, in both English and Japanese where it
   makes sense. Label them with what the **code policy** should produce, not what the model happens to say.
3. Run the unit tests (offline, fake client):
   ```bash
   pytest tests/test_llm_harness.py tests/test_exception_translator.py -q
   ```
   `test_eval_labels_match_code_policy` checks that new labels agree with the guardrails.
   `test_rules_baseline_accuracy_is_tracked` pins the rule parser's score. If you improve the rules,
   update that number in the same change.
4. LLM eval. It calls Claude about 30 times (a few cents with Claude Opus 5.5). **Ask the user before running
   it**, then:
   ```bash
   python -m ai_harness.evals.run_alert_eval --mode llm --out ai_harness/evals/results_llm.json
   ```
   Compare per-field accuracy and the failure list against the previous results file. Report regressions
   case by case; never report only the headline number.

## Adding a case

One JSON object per line in `ai_harness/evals/alert_cases.jsonl`:

```json
{"id": "jp-cap-window", "text": "受電上限を13時〜16時の間450kWに制限",
 "expected": {"soc_min": [], "grid_cap": [[13, 16, 450]], "end_soc_min_kwh": null},
 "tags": "japanese window cap"}
```

Every case uses a 24 h horizon starting at 00:00 and a 2,000 kWh battery. Windows are inclusive hour
indices. Values are kWh for reserves and kW for caps.
