# Bridging the English-Arabic Medical Knowledge Gap

This repository provides the code for **TLoRA (Targeted Low-Rank Adaptation)**, described in the paper ["Bridging the English-Arabic Medical Knowledge Gap: Targeted Low-Rank Adaptation via Causal Layer Selection"](https://arxiv.org/abs/2608.00207v1), accepted to Findings of EMNLP 2026. We show that Arabic medical knowledge is present in intermediate LLM representations but fails to surface at the output, and use mechanistic interpretability methods to localize this failure to a specific layer window. TLoRA restricts LoRA adaptation to that window, outperforming full-network LoRA and other baselines on Arabic medical QA. We also introduce **AraClinicDialog**, a clinician-constructed Arabic medical dialogue benchmark in MSA with validated variants across four Arabic dialects.

## Installation

Clone the repo:

```
git clone https://github.com/cha-y-mae/Adaptation.git
cd Adaptation
```

Create a virtual environment (Python 3.<X> or above) and install the requirements:

```
conda create -n adaptation python=3.<X> && conda activate adaptation
pip install -r requirements.txt
```

<!-- TODO: fill in the minimum Python version, and add any HF/API token setup instructions here if scripts require huggingface-cli login or an API key (e.g. for GPT-4o / Claude / Gemini baselines). -->

## Repository structure

```
Adaptation/
├── configs/      # experiment config files (model, layer window, learning rate, etc.)
├── datasets/     # dataset files 
├── diagnosis/    # tuned lens probing, causal activation patching, KL-divergence profiling scripts 
├── evals/        # evaluation scripts and metrics
├── models/       # model scripts for TLoRA and baselines
├── prompts/      # system prompts used for each task
├── scripts/      # entry-point scripts for running the pipeline
├── results/      # generated outputs 
└── requirements.txt
```

## Usage



### 1. Mechanistic diagnosis (tuned lens + causal activation patching)

```
python diagnosis/<script>.py --config configs/<diagnosis_config>.yaml
```

### 2. Train TLoRA

```
python scripts/<train_script>.py --config configs/<tlora_config>.yaml
```

### 3. Evaluate

```
python evals/<eval_script>.py --config configs/<eval_config>.yaml --task {mcqa,generation,dialogue}
```

## Reproducing results

All headline numbers reported in the paper (Tables 1–3 and appendix Tables S18–S24) can be reproduced by running the relevant config.

## AraClinicDialog

AraClinicDialog is our clinician-constructed Arabic medical dialogue benchmark, released in `datasets/`. <!-- TODO: confirm exact path/access instructions, and add license/usage terms for the dataset itself if different from the code license. -->

## Citing this work

```bibtex
@inproceedings{abouzahir2026bridging,
      title={Bridging the {E}nglish-{A}rabic Medical Knowledge Gap: Targeted Low-Rank Adaptation via Causal Layer Selection},
      author={Abouzahir, Chaimae and Khan, Musa and Ali-Hassan, Hala and Ma, Congbo and Saleh, Khaled and Sadqi, Yousra and Mallat, Jihad and Al-Eisawi, Walid and Habash, Nizar and Shamout, Farah E.},
      booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
      year={2026},
      url={https://openreview.net/forum?id=GLWhomy55Q},
}
```
## Contact

Please direct any questions to ca2627@nyu.edu.
