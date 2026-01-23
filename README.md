# soc-evaluation

Evaluation of the [Synthetic Online Conversations](https://huggingface.co/datasets/marcodsn/SOC-2508) dataset, specifically targeting Lexical Diversity.

## Exam-specific information (To Be Removed After the Exam)

The project document filename is: `main.html`  
The presentation filename is: `slides.html`  

The main raw dataset is located in the `external/` folder as a submodule.  
The pre-processed dataset and the generated complementary datasets are located in the `data/` folder.

## How to get started

> [!NOTE] 
> The dataset has been preprocessed already and it is now directly saved in this GitHub repository (at `data/processed/data.csv`). You do not need to follow the steps below to download the original dataset, unless you want to reprocess it from scratch.

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

## References

```bibtex
@misc{marcodsn_2025_SOC2508,
  title     = {Synthetic Online Conversations},
  author    = {Marco De Santis},
  year      = {2025},
  month     = {August},
  url       = {https://huggingface.co/datasets/marcodsn/SOC-2508},
}
```
