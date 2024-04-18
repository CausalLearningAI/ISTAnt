# ISTAnt

Automatic treatment effect estimation on ecological data with partial labelling

## Dataset description

### Data structure

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

### Data Distribution

![Outcome distribution](results/instant_lq/outcome_distribution.png)

### Research Question

Identify and estimate:
$$ATE_{B} := \mathbb{E}[Y|do(T=1)]- \mathbb{E}[Y|do(T=0)]$$
$$ATE_{INF} := \mathbb{E}[Y|do(T=2)]- \mathbb{E}[Y|do(T=0)]$$
