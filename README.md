# Language-Aware Adaptation of General-Purpose LLMs for Arabic Medical Tasks

This repository contains the experimental codebase for running and evaluating
LLM adaptation experiments on multiple tasks (mcq and answer generation),
driven by yaml configuration files.

## 1. Environment Setup

This project uses **Conda**.
**The Conda environment must be activated before running any scripts.**

### Create the environment

```bash
conda create -n adaptation python=3.10
```

### Activate the environment

```bash
conda activate adaptation
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Repository Structure (Relevant Parts)

Adaptation/
├── configs/                # YAML configs per task & model
│   ├── task1/
│   │   └── gpt5.yaml
│   └── task2/
│       └── gpt5.yaml
│
├── datasets/
│   └── qa/                 # task 1&2 datasets 
│       ├── arastem.json
│       ├── medarabench.json
│       ├── medarabiq.json
│       ├── mmlu-arabic.json
│
├── evals/                  # evaluation scripts
│   ├── evaluator.py
│   ├── metrics.py
│   └── __init__.py
│
├── models/                 # model handlers / wrappers
│   ├── llama70.py
│   ├── meditron70_handler.py
│   ├── openai_handler.py
│   └── __init__.py
│
├── prompts/                # Prompt templates per task
│   ├── task1.txt
│   └── task2.txt
│
├── scripts/                # entry-point scripts
│   ├── run_evaluation.py
│   ├── utils.py
│   └── __init__.py
│
├── results/
│   ├── metrics/            # aggregated evaluation results
│   │   ├── task1/
│   │   └── task2/
│   └── predictions/        # model outputs/predictions
│       ├── task1/
│       └── task2/
└── README.md

## 3. Running an Experiment

### Set API Keys

Make sure the required API keys are set as environment variables:

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### Configure the Experiment

Configuration files are stored under the `configs/` directory.
Each config defines:

* model type and model name
* dataset paths
* prompt/task settings
* caching behavior


### Run the Experiment

Always run experiments **from the root directory of the repository**. Same command below: 

```bash
python scripts/run_experiment.py configs/task1/gpt5.yaml
```

