# ISTAnt

Automatic treatment effect estimation on ecological data with partial labelling

## Dataset description

### Data structure

### Example

<div style="display:flex; justify-content:center; align-items:center;">
  <div style="margin-right:10px;">
    <p><i>No Action</i></p>
    <img src="img/example_nogrooming.gif" alt="grooming" width="200">
  </div>
  <div style="margin-left:10px;">
    <p><i>Grooming (Yellow)</i></p>
    <img src="img/example_grooming.gif" alt="nogrooming" width="200">
  </div>
</div>

### Data Distribution

![Outcome distribution](results/outcome_distribution.png)

### Research Question

Identify and estimate:
$$ATE_{B} := \mathbb{E}[Y|do(T=1)]- \mathbb{E}[Y|do(T=0)]$$
$$ATE_{INF} := \mathbb{E}[Y|do(T=2)]- \mathbb{E}[Y|do(T=0)]$$
