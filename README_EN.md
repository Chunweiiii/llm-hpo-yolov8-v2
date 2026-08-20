# LLM-Guided Hyperparameter Optimization for Traffic Light Detection

## What this is

We're testing whether an LLM can tune YOLOv8 hyperparameters better than standard search methods, by reading training diagnostics (loss curves, plateau detection, per-class weaknesses, etc.) and proposing changes each round. Traffic light detection is just the task we use to test this. We picked a hard version of the task on purpose (small objects, class imbalance, competing loss terms, mixed-source data) so there's actual room for optimization to matter.

Target: MDPI Applied Sciences, submitting early September.

## What we ran

Six methods, each given 10 rounds to tune 9 hyperparameters: Random Search, Optuna (TPE), and three LLMs (Qwen2.5-Coder-7B, Claude Haiku 4.5, Claude Sonnet 5, Claude Opus 5). We ran this twice — once on our original 727-image dataset, and again on a rebuilt 937-image dataset (cleaner, harder task, lower baseline score).

## Results

**Original dataset (727 images):** Claude Sonnet 5 won clearly — highest score (0.7187), zero duplicate proposals, used all 9 hyperparameters. Verified with 3 seeds: 0.7129 ± 0.0078, a real improvement over baseline.

**Rebuilt dataset (937 images):** Opus 5 scored highest (0.6766), but Random Search was close behind (0.6730). We verified both with 3 seeds — Opus: 0.6683 ± 0.0091, Random: 0.6633 ± 0.0080. **The gap between them (0.0050) is smaller than our noise threshold, so statistically they're tied.** LLM optimization no longer clearly beats random search on the harder dataset.

The other four methods (Haiku, Optuna, Sonnet, Qwen) only have single-run scores on the new dataset, not yet verified with multiple seeds.

## Why this matters for the paper

The original dataset told a clean story (LLM wins). The new dataset doesn't. We think this is worth discussing honestly rather than hiding — possible angles: the LLM advantage may shrink as task difficulty increases, or the LLM's real advantage might be reaching a good result in fewer rounds (Haiku/Optuna/Sonnet all peaked at round 1, vs. round 7 for Random/Opus — though this isn't verified yet). One thing that *does* hold up across both datasets: Qwen2.5-Coder-7B is consistently the weakest method with the highest duplicate-proposal rate.

## Still open

- Need to double check Optuna's round-0 baseline hyperparameters match the other methods' baseline exactly
- Haiku/Optuna/Sonnet/Qwen still need multi-seed verification on the new dataset if we want to use the "fewer rounds needed" story
