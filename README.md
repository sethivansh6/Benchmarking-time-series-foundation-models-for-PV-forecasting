# Benchmarking Time Series Foundation Models for PV Forecasting

A benchmarking study comparing zero-shot time-series foundation models (TSFMs), a tabular foundation model, and a LoRA fine-tuned variant against classical ML baselines for 24-hour-ahead PV power forecasting.

---

## Results

Evaluated on daytime samples only (solar elevation > 5°), 15-minute resolution, 24h horizon.

| Model | Type | nRMSE |
|---|---|---|
| LightGBM | Baseline | 0.041 |
| MLP | Baseline | 0.040 |
| TabPFN | Tabular FM (zero-shot) | 0.104 |
| TimesFM | TSFM (zero-shot) | 0.208 |
| Chronos (LoRA) | TSFM (fine-tuned) | 0.260 |
| Chronos | TSFM (zero-shot) | 0.270 |
| Moirai | TSFM (zero-shot) | 0.380 |

Feature-engineered baselines (LightGBM/MLP) significantly outperform all zero-shot TSFMs in this setting, largely because they receive measured irradiance and solar geometry as direct inputs while TSFMs operate univariate (power history only). TabPFN bridges the gap by taking the same weather features as the baselines without any training (nRMSE 0.104 vs 0.040). Among pure TSFMs, TimesFM performs best (0.208). LoRA fine-tuning of Chronos yields a modest 3.6% nRMSE improvement over its zero-shot baseline (0.270 → 0.260) in 5 epochs on CPU. Moirai underperforms the other TSFMs and also produces non-zero nighttime predictions, motivating the daytime-only evaluation metric.

---

## Models

- **LightGBM / MLP** — feature-engineered baselines using solar geometry + irradiance features
- **Chronos-T5-Small** — T5-based TSFM (~46M params), zero-shot and LoRA fine-tuned
- **TimesFM** — Google patched-decoder TSFM, zero-shot
- **Moirai-Small** — Salesforce unified TSFM, zero-shot
- **TabPFN v2** — PriorLabs tabular foundation model, zero-shot via API

---

## Repository Structure

```
├── notebooks/          # Numbered Jupyter notebooks (01 → 08, run in order)
├── data/               # Raw + processed CSVs (gitignored)
├── figure/             # Generated plots
├── models/
│   └── chronos_lora_adapter/   # Saved LoRA adapter weights
├── results/            # Prediction .pkl files and metrics
```

---

## Setup

### Base Environment

Works for notebooks 01–04, 06–08 (Chronos, TimesFM, baselines, TabPFN, LoRA).

```bash
conda create -n pv_bench python=3.11
conda activate pv_bench

pip install torch==2.1.0
pip install chronos-forecasting==1.3.0
pip install timesfm==1.0.0
pip install lightgbm==4.3.0
pip install peft==0.9.0
pip install tabpfn
pip install scikit-learn pandas numpy pvlib matplotlib seaborn jupyter
```

### Moirai — Separate Environment

`uni2ts` has dependency conflicts with Chronos and TimesFM. Use a separate environment for notebook 05 only.

```bash
conda create -n moirai_env python=3.11
conda activate moirai_env

pip install torch==2.1.0
pip install "uni2ts==1.1.0"
pip install pandas numpy pvlib jupyter
```

### TabPFN API Token

TabPFN v2 requires a free API token from PriorLabs.

1. Sign up at [https://ux.priorlabs.ai](https://ux.priorlabs.ai)
2. Go to **Account → API Keys** and generate a token
3. Export before running notebook 06:

```bash
export TABPFN_API_KEY="your_token_here"
```

Or set it at the top of the notebook:

```python
import os
os.environ["TABPFN_API_KEY"] = "your_token_here"
```

> Do not commit your API key.

---

## Reproducing the Benchmark

```bash
conda activate pv_bench
jupyter notebook
```

Run notebooks in order: **01 → 02 → 03 → 04 → 05 → 06 → 07 → 08**  
For notebook 05 (Moirai), switch to `moirai_env` first, then switch back.

Notebook 08 aggregates all saved predictions from `results/` and regenerates every figure in `figure/`.

---

## Dependencies

| Package | Version |
|---|---|
| Python | 3.11 |
| PyTorch | 2.1.0 |
| chronos-forecasting | 1.3.0 |
| timesfm | 1.0.0 |
| uni2ts (Moirai) | 1.1.0 |
| tabpfn | 2.0 |
| peft | 0.9.0 |
| lightgbm | 4.3.0 |
| pvlib | 0.10.x |

All experiments run on CPU — no GPU required.
