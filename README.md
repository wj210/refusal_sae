## Understanding Refusal in Language Models with Sparse Autoencoders (Accepted to EMNLP 2025!)

## Setup

Install the required packages via `pip install -r requirements.txt`.

Benchmark harmful and harmless datasets are taken from https://github.com/andyrdt/refusal_direction. Download the processed, raw and splits and put inside a `dataset` folder.

## How to run

All experiments can be ran in notebook style, which is the preferred manner. 

Benchmark on harmful datasets and capabilities are ran with `benchmark.ipynb`

Section 4.2 is ran with `cat_harm.ipynb`

Section 4.3 and OOD probing is ran with `wildjailbreak.ipynb`

If you prefer to run in normal `.py` format, the benchmark scores can be retrieved by running
```python
python -m src.benchmark --model gemma-2b --bz 32 --la_bz 5 --use_vllm True --eval_jb True
```

and CATQA scores by running
```python
python -m src.cat_harm --model gemma-2b --bz 32 --la_bz 5 --use_vllm True --eval_jb True
```

## Memory requirements

We use vllm (https://github.com/vllm-project/vllm) to load the harmbench classifier. If you have issues with running vllm, you can default to regular `transformers` and set `--use_vllm False` in the `src` files and also the function `load_harmbench_classifier(use_vllm=False)` in the notebooks.

The batch sizes are tailored for 80GB GPUs, adjust accordingly.

If you do run the notebooks or python files, we ran it with 2 GPU, since we are loading 2 models (base model and harmbench classifier) and implementing attribution patching (LA-IG) at the same time, which requires computing gradients. This would not fit within a single GPU. 

If you only have access to a single GPU. For the `src` files, you should first run 
```python
python -m src.get_features
```
to get the feature set before running `benchmark.py` or `cat_harm.py`. This would first do LA-IG and cache the features.

If you only have access to GPU with lower VRAM, you can eval the scores separately by setting `--eval_jb False` in the src files and retrieve the scores via
```python
python -m src.eval_scores --model gemma-2b --bz 32 --dataset benchmark
```
So the logic becomes `get_features` -> `benchmark/cat_harm` -> `eval_scores` -> `plot`, where `plot` produces the figures.

Set the `--dataset` to either `benchmark` (Section 4.1) or `cat_harm` for section 4.2

## Libraries

We use transformer_lens (https://github.com/TransformerLensOrg/TransformerLens) for the LLMs and SAE_lens (https://github.com/jbloomAus/SAELens) for SAEs. We do not use NNSight as we had issues implementing attribution patching (taken from https://github.com/saprmarks/feature-circuits) on SAEs.

## Citation
Please cite our work if you found it useful! 

```bibtex
@article{yeo2025understanding,
  title={Understanding Refusal in Language Models with Sparse Autoencoders},
  author={Yeo, Wei Jie and Prakash, Nirmalendu and Neo, Clement and Lee, Roy Ka-Wei and Cambria, Erik and Satapathy, Ranjan},
  journal={arXiv preprint arXiv:2505.23556},
  year={2025}
}
