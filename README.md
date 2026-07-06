# Celo2: Towards Learned Optimization Free Lunch

> **Abhinav Moudgil, Boris Knyazev, Eugene Belilovsky**   
> ICLR 2026   
> https://arxiv.org/abs/2602.19142    

A self-contained single-file Optax implementation of Celo2.

Celo2 is a simple learned MLP update rule that can be meta-trained in a few GPU hours and scales stably to out-of-distribution tasks much larger than its meta-training distribution (tested up to GPT-3 1.3B). We release pretrained optimizer weights as well as support for meta-training.

## Installation

```bash
pip install git+https://github.com/amoudgl/celo2.git
```
or simply copy `celo2_optax.py` into your project and go.

## Download

Pretrained optimizer weights are available on HuggingFace and can be downloaded via commands below with [CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) tool:

| Optimizer | HuggingFace | Download command
| -------- | ------- | ------- |
| celo2 | [repo](https://huggingface.co/amoudgl/celo2) | `hf download amoudgl/celo2 --local-dir ./celo2`
| celo2-base | [repo](https://huggingface.co/amoudgl/celo2-base) | `hf download amoudgl/celo2-base --local-dir ./celo2-base`

**Celo2 vs Celo2-base.** Celo2 applies Newton-Schulz orthogonalization on top of the learned MLP update rule for matrix (2D) parameters from hidden layers and uses AdamW for biases/embedding parameters. Celo2-base uses the learned update rule for all parameters. Both have been meta-trained on 4 simple image MLP classification tasks from [Celo](https://arxiv.org/abs/2501.12670) but work out-of-the-box stably on unseen tasks in our experiments. We recommend Celo2 for practical use and better performance. See the [paper](https://arxiv.org/abs/2602.19142) for details.

## Usage

[Example: language model pretraining with Celo2](https://github.com/amoudgl/batch-size/blob/celo2/pretraining/optimizer.py#L287-L309)

Our `celo2_optax` package exposes `scale_by_celo2`, an `optax.GradientTransformation` that applies the learned MLP update rule, and `load_checkpoint` utility method for loading meta-trained optimizer weights from a path.

Compose an Optax transform with `scale_by_celo2` like any standard optimizer:
```python
import optax
from celo2_optax import scale_by_celo2, load_checkpoint

# celo2
pretrained_params = load_checkpoint('path/to/checkpoint')
scaled_lr_schedule = lambda step: mult_1d * lr_schedule(step)
optimizer = optax.multi_transform(
    transforms={
        'celo2': optax.chain(
            scale_by_celo2(pretrained_params, orthogonalize=True),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_learning_rate(lr_schedule),
        ),
        'adam': optax.adamw(scaled_lr_schedule, 0.9, 0.95, weight_decay=weight_decay)
    },
    # just an example, define param_labels function as per your task
    param_labels=lambda params: jax.tree.map_with_path(
        lambda path, val: 'adam' if val.ndim <= 1 or 'embed' in jax.tree_util.keystr(path) else 'celo2', params
    ),
)

# standard optax use after declaring optimizer
opt_state = optimizer.init(params)
updates, opt_state = optimizer.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

To try celo2-base, do:
```python
import optax
from celo2_optax import scale_by_celo2, load_checkpoint

# celo2-base
pretrained_params = load_checkpoint('path/to/checkpoint')
optimizer = optax.chain(
    scale_by_celo2(pretrained_params, orthogonalize=False),
    optax.add_decayed_weights(weight_decay),
    optax.scale_by_learning_rate(lr_schedule),
)
```

**Configuration.** See `Celo2Transformation.__init__` for the full set of options. Defaults are set for Celo2. The only difference between Celo2 and Celo2-base configuration is the `orthogonalize` flag: set `True` for Celo2 (with AdamW for 1D params via `optax.multi_transform`), `False` for Celo2-base.

## Meta-training

**Setup.** Meta-training runs from the [celo](https://github.com/amoudgl/celo) repository. Quick install:
```
git clone git@github.com:amoudgl/celo.git
cd celo
uv sync --active
source .venv/bin/activate
```
Optionally, set `TFDS_DATA_DIR` to download and setup meta-training datasets at a custom location; otherwise the meta-training script uses tensorflow's default cache directory:
```
export TFDS_DATA_DIR=/path/to/tensorflow_datasets
```

**Code layout.** In `celo` repository, `celo/optimizers/celo2.py` is simply a wrapper around `celo2_optax.py` that allows integration with `learned_optimization` package to support meta-training. The core learned MLP update and optax transformations live in `celo/optimizers/celo2_optax.py`, which matches the self-contained `celo2_optax.py` in this repo.

**Run.** To meta-train Celo2 on the 4 small image MLP classification tasks as in the original work, run the command below:
```bash
python -m celo.train --optimizer=celo2 --exp_name=celo2 --outer_iterations=100000 --max_unroll_length=2000 --seed=0 --task=fast_velo --outer_lr=0.00005 --aug=reparam --aug_reparam_level=global --trainer=pes --step_mult=0.001 --experiment_root=~/celo_experiments --exp_id=celo2 --regex_1d=/b$
```

Note that `--regex_1d` is a Python regex for Celo2 on flattened parameter paths: leaves whose path matches get the AdamW branch; everything else uses the learned Celo2 update during meta-training. Specify it correctly as per your meta-training task. The command above uses regex `/b$` that matches bias parameters in the 4 image classification tasks used in Celo2 meta-training (bundled as `fast_velo`).

For celo2-base, do:
```bash
python -m celo.train --optimizer=celo2base --exp_name=celo2base --outer_iterations=100000 --max_unroll_length=2000 --seed=0 --task=fast_velo --outer_lr=0.0001 --aug=reparam --aug_reparam_level=global --trainer=pes --step_mult=0.001 --experiment_root=~/celo_experiments --exp_id=celo2base
```

Meta-training should finish in <6h for both variants on a single A100 GPU.

`--exp_name` is the run name used in Weights & Biases when logging is enabled; `--exp_id` is the subdirectory name under `<experiment_root>/train/` where checkpoints, config, and metrics are written (if unset, an id is auto-generated). For more on training flags, see the [training script](https://github.com/amoudgl/celo/blob/main/celo/train.py) in the [celo](https://github.com/amoudgl/celo) repository.

## Sharp edges
- The implementation uses last two dimensions of matrix parameters by default for orthogonalization and RMS normalization (look [here](https://github.com/amoudgl/celo2/blob/main/celo2_optax.py#L461) and [here](https://github.com/amoudgl/celo2/blob/main/celo2_optax.py#L64-L81)): if a parameter has more than two dimensions, the implementation treats the last two as the matrix to operate on and the leading dimensions are batched (parallelized) over. If your evaluation task's network doesn't have this structure by default, i.e. parameter dimensions that should be orthogonalized/normalized are not the last two, you'll need to either adapt the task network implementation or modify the Celo2 forward pass.
- We use the LM-30M task (see [paper](https://arxiv.org/abs/2602.19142) for details of this task) as the validation task: we evaluate final optimizer checkpoint from each meta-training sweep run and pick the one that performs the best on the LM-30M task. We do this because of the large gap between meta-training (image classification with small MLPs) and evaluation tasks (million/billion-scale language modeling with transformers) which leads to generalization issue -- the optimizer checkpoint with lowest meta-training loss may not always perform the best on downstream evaluation tasks, and we found that LM-30M works well as a proxy.

## Citation

```bibtex
@misc{moudgil2026celo2,
      title={Celo2: Towards Learned Optimization Free Lunch},
      author={Abhinav Moudgil and Boris Knyazev and Eugene Belilovsky},
      year={2026},
      eprint={2602.19142},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.19142},
}
```

## License

MIT
