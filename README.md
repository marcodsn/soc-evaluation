# soc-evaluation

Evaluation of the Synthetic Online Conversations dataset, specifically targeting Lexical Diversity.

## How to get started

> [!NOTE] 
> The dataset has been preprocessed already and it is now directly saved in this GitHub repository (at data/processed/data.csv). You do not need to follow the steps below to download the original dataset, unless you want to reprocess it from scratch.

This repository contains datasets as submodules; to download the repo correctly, including the datasets, use the recursive clone option:

```bash
git clone --recursive https://github.com/marcodsn/soc-evaluation.git
```

Or initialize submodules after cloning:

```bash
git clone https://github.com/marcodsn/soc-evaluation.git
cd soc-evaluation

git submodule init
git submodule update
```
