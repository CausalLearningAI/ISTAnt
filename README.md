# ISTAnt

Prediction-powered causal inference on high-dimensional data pipeline for automatic treatment effect estimation, with a specific showcase on a new ecological dataset, [ISTAnt](https://doi.org/10.6084/m9.figshare.26484934.v2), to study social immunity in ant colonies.

## Get started

### Requirements

Using Conda: 
```conda
conda env create -f environment.yml
```

### Dataset

#### Download
Download the dataset from [Figshare](https://doi.org/10.6084/m9.figshare.26484934.v2) and place it in the foldere `ISTAnt/data/`.

#### Preview
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

#### Description
ISTAnt is a new ecological dataset for social immunity and represents the first real-world benchmark for causal inference downstream tasks on high-dimensional observations. It analyzes grooming behavior in the ant Lasius neglectus in groups of three worker ants. The workers for the experiment were obtained from their laboratory stock colony, which had been collected from the field in 2022 in the Botanical Garden Jena, Germany. Ant collection and all experimental work were performed in compliance with international, national and institutional regulations and ethical guidelines. For the experiment, the body surface of one of the three ants was treated with a suspension of either of two microparticle types (diameter ~5 µm) to induce grooming by the two nestmates, which were individually color-coded by application of a dot of blue or orange paint, respectively. The three ants were housed in small plastic containers (diameter 28mm, height 30mm) with moistened, plastered ground and the interior walls covered with PTFE (polytetrafluoroethane) to hamper climbing by the ants. Filming occurred in a temperature- and humidity-controlled room at 23°C within a custom-made filming box with controlled lighting and ventilation conditions. We set up nine ant groups at a time (always containing both treatments) and placed them randomly on positions 1-9 marked on the floor in a 3x3 grid, about 3mm from each other. The experiment was performed on two consecutive days. Videos were acquired using a USB camera (FLIR blackfly S BFS-U3-120S4C, Teledyne FLIR) with a high-performance lens (HP Series 25mm Focal Length, Edmund optics 86-572) in OBS studio 29.0.0 \citep{bailey2017obs} at a framerate of 30 FPS and a resolution of 2500x2500 pixels. From each original video (105x105 mm), we generated nine individual videos .mkv (each ~32x32 mm, 770x770 pixels) by determining exact coordinates per container from one frame in GIMP 2.10.36 and cropping of the videos with FFmpeg 6.1.1. Annotation was performed over two consecutive days by three observers who had not been involved in the experimental setup or recording and were unaware of the treatment assignments to ensure bias-free behavioral annotation. They annotated the behavior of the ants during video observations, using custom-made software that saves the start and end frames of behaviors marked in a .csv file (see 'annotations' folder). In one of the videos, one of the nestmates' legs got inadvertently stuck to its body surface during the color-coding, interfering with its behavior, so the video was discarded. This left 44 videos from 5 independent setups (n=24 of treatment 1 and n=20 of treatment 2) of 10 minutes each for a total of 792 000 annotated frames (see 'video' folder). For each video, we provide the following information: the number of the set to which it belongs (1-5); the number of the position within the set reflecting the position of the ant group under the camera (1-9), for which we also provide ‘coordinates’ in the 3x3 grid (taking values -1/0/1 for both X and Y axis); treatment (1 or 2); the hour of the day when the recording was started (in 24h CEST); experimental day (A or B); the top left coordinate of the cropping square from the original video (CropX/CropY); the person annotating the video (given as A, B, C); the date of annotation (1: first day, 2: second day) and in which order the videos were annotated by each person, both reflecting a possible training effect of the person (see 'experiments_settings.csv' file).


## Research Question

Identify and estimate the Average Treatment Effect:

$$ATE := \mathbb{E}[Y|do(T=2)]- \mathbb{E}[Y|do(T=1)]$$.

## Analyses

### Biases Investigation

_Experiments (run)_: [run_experiments.py](https://github.com/CausalLearningAI/ISTAnt/blob/main/src/run_experiments.py)

_Experiments (visualize)_: [biases.ipynb](https://github.com/CausalLearningAI/ISTAnt/blob/main/experiments/biases.ipynb)

_Reference_: [Smoke and Mirrors in Causal Downstream Tasks](https://arxiv.org/abs/2405.17151)

```bibtex
@article{cadei2024smoke,
  title={Smoke and Mirrors in Causal Downstream Tasks},
  author={Cadei, Riccardo and Lindorfer, Lukas and Cremer, Sylvia and Schmid, Cordelia and Locatello, Francesco},
  journal={arXiv preprint arXiv:2405.17151},
  year={2024}
}
```

Further analyses on a synthetic data set (CausalMNIST) are reported at [https://github.com/CausalLearningAI/CausalMNIST](https://github.com/CausalLearningAI/CausalMNIST).

### Enforcing Experiment Setting Invariance

_Experiments Notebook:_ [invariance.ipynb](https://github.com/CausalLearningAI/ISTAnt/blob/main/experiments/invariance.ipynb)

_Reference:_ [Unifying Causal Representation Learning with the Invariance Principle](https://www.arxiv.org/abs/2409.02772)

```bibtex
@article{yao2024unifying,
  title={Unifying Causal Representation Learning with the Invariance Principle},
  author={Dingling, Yao and Rancati, Dario and Cadei, Riccardo and Fumero, Marco and Locatello, Francesco},
  journal={arXiv preprint arXiv:2409.02772},
  year={2024}
}
```
