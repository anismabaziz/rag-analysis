# Retrieval results

3 runs at commit 2a195cdf66b3 against corpus sha256:4d4ee4bb36386c91f7118ea16f5f1491acbe34d2770b6b7da122aece313efc08.
Frozen configuration: prompt sha256:351044bda5e768c92fd92ba80abfc2d93179727867ac8b3e652959a76429289a, qwen/qwen3.8-27b at temperature 0.0, depth 5.

| architecture | scope | questions | scored | recall@1 | recall@3 | recall@5 | ndcg@1 | ndcg@3 | ndcg@5 | mrr | cite | cite_answerable | cite_unanswerable | exact_match | exact_match_n | cite_extractive | retrieval_p50_s | retrieval_p95_s | generation_p50_s | generation_p95_s | tokens_total | tokens_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid | papers | 32 | 28 | 0.8036 | 0.8571 | 0.8571 | 0.8214 | 0.8411 | 0.8411 | 0.8554 | 0.0375 | 0.0429 | 0.0000 | 0.0000 | 16 | 0.0625 | 0.0704 | 0.0837 | 14.6422 | 46.8713 | 33153 | 1036.0312 |
| hybrid | papers:identifier_heavy | 14 | 14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0714 | 0.0714 | - | 0.0000 | 14 | 0.0714 | 0.0686 | 0.0826 | 14.0025 | 34.5207 | 14086 | 1006.1429 |
| hybrid | papers:paraphrase | 10 | 10 | 0.6000 | 0.6000 | 0.6000 | 0.6000 | 0.6000 | 0.6000 | 0.6452 | 0.0000 | 0.0000 | - | 0.0000 | 2 | 0.0000 | 0.0712 | 0.0897 | 14.2826 | 40.2269 | 11126 | 1112.6000 |
| hybrid | pooled | 32 | 28 | 0.8036 | 0.8571 | 0.8571 | 0.8214 | 0.8411 | 0.8411 | 0.8554 | 0.0375 | 0.0429 | 0.0000 | 0.0000 | 16 | 0.0625 | 0.0704 | 0.0837 | 14.6422 | 46.8713 | 33153 | 1036.0312 |
| hybrid | pooled:identifier_heavy | 14 | 14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0714 | 0.0714 | - | 0.0000 | 14 | 0.0714 | 0.0686 | 0.0826 | 14.0025 | 34.5207 | 14086 | 1006.1429 |
| hybrid | pooled:paraphrase | 10 | 10 | 0.6000 | 0.6000 | 0.6000 | 0.6000 | 0.6000 | 0.6000 | 0.6452 | 0.0000 | 0.0000 | - | 0.0000 | 2 | 0.0000 | 0.0712 | 0.0897 | 14.2826 | 40.2269 | 11126 | 1112.6000 |
| naive | papers | 32 | 28 | 0.8393 | 0.8571 | 0.9643 | 0.8571 | 0.8571 | 0.9033 | 0.8839 | 0.0160 | 0.0183 | 0.0000 | 0.0000 | 16 | 0.0000 | 0.0410 | 0.0833 | 9.2254 | 36.6950 | 25438 | 794.9375 |
| naive | papers:identifier_heavy | 14 | 14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | - | 0.0000 | 14 | 0.0000 | 0.0389 | 0.0624 | 8.3129 | 29.3930 | 10660 | 761.4286 |
| naive | papers:paraphrase | 10 | 10 | 0.6000 | 0.6000 | 0.9000 | 0.6000 | 0.6000 | 0.7292 | 0.6750 | 0.0000 | 0.0000 | - | 0.0000 | 2 | 0.0000 | 0.0372 | 0.0940 | 7.3879 | 34.7596 | 8807 | 880.7000 |
| naive | pooled | 32 | 28 | 0.8393 | 0.8571 | 0.9643 | 0.8571 | 0.8571 | 0.9033 | 0.8839 | 0.0160 | 0.0183 | 0.0000 | 0.0000 | 16 | 0.0000 | 0.0410 | 0.0833 | 9.2254 | 36.6950 | 25438 | 794.9375 |
| naive | pooled:identifier_heavy | 14 | 14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | - | 0.0000 | 14 | 0.0000 | 0.0389 | 0.0624 | 8.3129 | 29.3930 | 10660 | 761.4286 |
| naive | pooled:paraphrase | 10 | 10 | 0.6000 | 0.6000 | 0.9000 | 0.6000 | 0.6000 | 0.7292 | 0.6750 | 0.0000 | 0.0000 | - | 0.0000 | 2 | 0.0000 | 0.0372 | 0.0940 | 7.3879 | 34.7596 | 8807 | 880.7000 |
| sparse | papers | 32 | 28 | 0.7321 | 0.8393 | 0.9107 | 0.7500 | 0.7944 | 0.8236 | 0.8077 | 0.0417 | 0.0476 | 0.0000 | 0.0000 | 16 | 0.0833 | 0.0400 | 0.0704 | 5.5078 | 15.4422 | 17128 | 535.2500 |
| sparse | papers:identifier_heavy | 14 | 14 | 0.9286 | 1.0000 | 1.0000 | 0.9286 | 0.9643 | 0.9643 | 0.9524 | 0.0952 | 0.0952 | - | 0.0000 | 14 | 0.0952 | 0.0435 | 0.0575 | 5.5497 | 24.9518 | 8261 | 590.0714 |
| sparse | papers:paraphrase | 10 | 10 | 0.5000 | 0.6000 | 0.8000 | 0.5000 | 0.5631 | 0.6448 | 0.5950 | 0.0000 | 0.0000 | - | 0.0000 | 2 | 0.0000 | 0.0200 | 0.0671 | 4.5935 | 7.2525 | 4628 | 462.8000 |
| sparse | pooled | 32 | 28 | 0.7321 | 0.8393 | 0.9107 | 0.7500 | 0.7944 | 0.8236 | 0.8077 | 0.0417 | 0.0476 | 0.0000 | 0.0000 | 16 | 0.0833 | 0.0400 | 0.0704 | 5.5078 | 15.4422 | 17128 | 535.2500 |
| sparse | pooled:identifier_heavy | 14 | 14 | 0.9286 | 1.0000 | 1.0000 | 0.9286 | 0.9643 | 0.9643 | 0.9524 | 0.0952 | 0.0952 | - | 0.0000 | 14 | 0.0952 | 0.0435 | 0.0575 | 5.5497 | 24.9518 | 8261 | 590.0714 |
| sparse | pooled:paraphrase | 10 | 10 | 0.5000 | 0.6000 | 0.8000 | 0.5000 | 0.5631 | 0.6448 | 0.5950 | 0.0000 | 0.0000 | - | 0.0000 | 2 | 0.0000 | 0.0200 | 0.0671 | 4.5935 | 7.2525 | 4628 | 462.8000 |

A scope of papers:identifier_heavy slices a domain to one stratum; pooled:identifier_heavy pools that slice across domains. The claim is a difference between those slices, so they are rows rather than a footnote.
cite is the share of answer claims one retrieved chunk each supports; cite_unanswerable is the same share over the unanswerable stratum alone.
exact_match is the share of extractive questions whose answer matches the gold span, tolerating surrounding whitespace and punctuation only; cite_extractive is citation accuracy over those same questions.
retrieval_p50_s and retrieval_p95_s are the median and 95th percentile retrieval seconds over the run's questions; generation_p50_s and generation_p95_s are the same over the generated answers alone. tokens_total is prompt plus completion words summed over the generated answers, counted as whitespace-separated words.
