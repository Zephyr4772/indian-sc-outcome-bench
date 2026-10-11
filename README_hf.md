---
license: openrail
language:
  - en
pretty_name: Indian Supreme Court Judgment Outcome Bench (recent vs old)
size_categories:
  - n<1K
task_categories:
  - text-classification
tags:
  - legal
  - india
  - supreme-court
  - benchmark
  - contamination
  - llm-evaluation
configs:
  - config_name: default
    data_files:
      - split: test
        path: data/test.jsonl
---

# Indian Supreme Court Judgment Outcome Bench

**380 Indian Supreme Court judgments.** The task: predict whether the appeal or petition was **allowed** or **dismissed**, given only the opening of the judgment (facts and arguments, with the decision removed).

The set is built to separate **reasoning from recall**:
- **Two halves with the same task and format:** 190 **recent** judgments (2023–24) and 190 **old** ones (1995–2005), both with the same 135 allowed / 55 dismissed split.
- **A second, name-only condition:** each case is also asked with only its title and date. A model that does better on old cases, or that can answer from the name alone, is remembering, not reasoning.

- **Code and raw predictions:** [github.com/Zephyr4772/indian-sc-outcome-bench](https://github.com/Zephyr4772/indian-sc-outcome-bench)
- **Source judgments:** [sinhal/Indian_Supreme_Court_Judgments](https://huggingface.co/datasets/sinhal/Indian_Supreme_Court_Judgments)

## Results: five open 7–8B models (zero-shot)

**Balanced accuracy** is the headline metric: chance and always-the-same-answer both score 50%. The raw majority baseline (always "allowed") is 71.1%.

**With judgment text:**

| model | bal. acc | old | recent | old − recent (95% CI) | answers "allowed" | AUROC |
|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | **55.4** | 54.2 | 56.5 | −2.3 [−12.0, +7.7] | 75.5% | n/a* |
| Qwen3-8B (no thinking) | 52.1 | 51.5 | 52.7 | −1.1 [−12.0, +9.7] | 42.1% | 0.58 |
| Qwen2.5-7B-Instruct | 51.6 | 49.7 | 53.6 | −4.0 [−15.3, +6.8] | 40.5% | 0.58 |
| Phi-4-mini-instruct | 51.1 | 51.4 | 50.8 | +0.7 [−6.3, +7.5] | 11.6% | 0.55 |
| Granite-3.3-8B-instruct | 49.4 | 49.4 | 49.3 | +0.1 [−6.6, +7.0] | 9.2% | 0.59 |

**Name and date only (recall probe):**

| model | bal. acc | old | recent | answers "allowed" | unparsed |
|---|---|---|---|---|---|
| Qwen3-8B | 50.4 | 50.3 | 50.5 | 84.2% | 0% |
| Qwen2.5-7B-Instruct | 50.0 | 50.0 | 50.0 | 100% | 0% |
| Phi-4-mini-instruct | 49.3 | 50.0 | 48.6 | 0.8% | 0% |
| Granite-3.3-8B-instruct | 47.1 | 42.5 | 51.8 | 26.3% | 1.6% |
| Mistral-7B-Instruct-v0.3 | 28.7 | 50.0 | 7.4 | 57.9% | 42.1% (recent cases only, see below) |

\*Mistral's first generated token is a bare space, so its option probabilities were not captured. Its answers are valid.

**Findings:**
1. **All five models are at chance.** Balanced accuracy is 49–55%, and the probability signal is faint (AUROC ≤ 0.59).
2. **Each model mostly applies a fixed bias instead of reading the case.** Granite and Phi answer "dismissed" ~90% of the time, while Mistral answers "allowed" 76% of the time. The true split is 71% / 29%.
3. **No memorisation is visible at this size.** Old and recent cases score the same for every model (every gap's confidence interval includes 0), and no model recalls outcomes from the case name.
4. **Mistral recognises dates beyond its training data.** Given only the name and date, it declined to answer for **160 of 190 recent** cases ("Pending…", "I'm an AI…") and for **0 of 190 old** ones. On old cases it is still at chance, so this is date-awareness, not recall.

**Setup:** zero-shot, greedy decoding, the same prompt for every model, vLLM 0.31 on Kaggle 2× T4 (fp16), max context 10,240 tokens. The longest prompt is 8,415 tokens with the Granite tokenizer, so nothing is truncated.

## How items were built (`build.py`)

- **Source:** English, full-length judgments (not short orders) from 1995–2005 and 2023–24. Duplicate PDFs listed under several years are removed.
- **Label:** kept only when the dataset's `outcome` field **and** the judgment's own closing paragraphs agree ("…appeal is allowed" / "…dismissed"). Partly-allowed, disposed-of and ambiguous cases are excluded. The metadata `outcome` contradicted the judgment text in **8% of old** and **3% of recent** candidates, which is why the cross-check matters.
- **Input:** page headers and signatures are stripped, then the text is cut at a paragraph boundary before the final 35%, capped at 14,000 characters. Any item whose kept text still states the outcome is dropped.
- **Pairing:** all 190 recent items that pass, plus a seeded random sample of old items with the same label counts.

## Fields

| field | description |
|---|---|
| `id` | `ISC-0000` … |
| `era` | `recent` (2023–24) or `old` (1995–2005) |
| `year`, `judgment_date` | decision date |
| `petitioner`, `respondent` | party names (used by the name-only probe) |
| `text` | opening of the judgment, decision removed (median ~10.5k characters) |
| `label` | `allowed` / `dismissed` |
| `source_id`, `pdf`, `full_text_chars` | link back to the source dataset |

## Limitations

- **Small:** 380 items, and the dismissed class has 110. Old − recent gaps under about ±10 points are within noise; the confidence intervals are reported for that reason.
- **Binary labels only.** Partly-allowed and disposed-of outcomes are excluded.
- **The input can still lean toward the outcome.** The opening may summarise the court's reasoning. Explicit outcome statements are filtered, but the input isn't a sanitised fact pattern.
- **"Recent" is relative.** 2023 cases may be inside newer models' training data. The name-only probe is the direct check.
- **English only,** because the source's vernacular texts have broken font encoding.

## Load

```python
from datasets import load_dataset

ds = load_dataset("sinhal/indian-sc-outcome-bench", split="test")
```

Submit your model's results in a discussion and I'll add it to the table.
