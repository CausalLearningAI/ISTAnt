# ISTAnt

Automatic treatment effect estimation on ecological data with partial labeling.

## Dataset

The data set can be preliminary inspected [here](https://drive.google.com/drive/folders/1ZTPusp-u3pAtrs2LtA3JUaFXbuqDS7K_?usp=sharing). We will publish a de-anonymized project page after the review process.

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


## Research Question

Identify and estimate:
$$ATE_{B} := \mathbb{E}[Y|do(T=1)]- \mathbb{E}[Y|do(T=0)]$$
$$ATE_{INF} := \mathbb{E}[Y|do(T=2)]- \mathbb{E}[Y|do(T=0)]$$
