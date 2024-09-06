# ISTAnt

Automatic treatment effect estimation on ecological data with partial labeling.

## Dataset description

The data set can be preliminarily inspected [here](https://figshare.com/s/0970e149cfe72089c771?file=48137317). After the review process, we will publish a de-anonymized project page.

### Example

<table align="center">
  <tr>
    <th>No Action</th>
    <th>Grooming (<i>Y2F</i>)</th>
  </tr>
  <tr>
    <td><img src="img/example_nogrooming.gif" alt="GIF 1" width="300" height="300"></td> 
    <td><img src="img/example_grooming.gif" alt="GIF 2" width="300" height="300"></td>
  </tr>
</table>

### Research Question


Identify and estimate:

$$ATE := \mathbb{E}[Y|do(T=1)]- \mathbb{E}[Y|do(T=0)]$$

## Analyses

### Biases Investigation

_Experiments (run)_: [run_experiments.py](https://github.com/CausalLearningAI/InvarianceCRL/blob/main/src/run_experiments.py)

_Experiments (visualize)_: [biases.ipynb](https://github.com/CausalLearningAI/InvarianceCRL/blob/main/experiments/biases.ipynb)

_Reference_: [Smoke and Mirrors in Causal Downstream Tasks](https://arxiv.org/abs/2405.17151)

```bibtex
@article{cadei2024smoke,
  title={Smoke and Mirrors in Causal Downstream Tasks},
  author={Cadei, Riccardo and Lindorfer, Lukas and Cremer, Sylvia and Schmid, Cordelia and Locatello, Francesco},
  journal={arXiv preprint arXiv:2405.17151},
  year={2024}
}
```

### Enforcing Experiment Setting Invariance

_Experiments Notebook:_ [invariance.ipynb](https://github.com/CausalLearningAI/InvarianceCRL/blob/main/experiments/invariance.ipynb)

_Reference:_ [Unifying Causal Representation Learning with the Invariance Principle](https://www.arxiv.org/abs/2409.02772)

```bibtex
@article{yao2024unifying,
  title={Unifying Causal Representation Learning with the Invariance Principle},
  author={Dingling, Yao and Rancati, Dario and Cadei, Riccardo and Fumero, Marco and Locatello, Francesco},
  journal={arXiv preprint arXiv:2409.02772},
  year={2024}
}
```