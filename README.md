# Heuristic-free Optimization of Force-Controlled Robot Search Strategies in Stochastic Environments

This repository contains the code for ∂PSE, a framework for the data-driven optimization of robot programs in stochastic environments.

∂PSE was presented in our paper "Heuristic-free Optimization of Force-Controlled Robot Search Strategies in Stochastic Environments", presented at the 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) in Kyoto, Japan.

ArXiv: https://arxiv.org/abs/2207.07524

IEEE Xplore: https://ieeexplore.ieee.org/document/9982093

To cite ∂PSE, consider including the following BibTeX in your bibliography:

```
@inproceedings{alt_heuristic-free_2022,
  title = {Heuristic-Free Optimization of Force-Controlled Robot Search Strategies in Stochastic Environments},
  booktitle = {2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  author = {Alt, Benjamin and Katic, Darko and Jäkel, Rainer and Beetz, Michael},
  year = {2022},
  pages = {8887--8893},
  publisher = {IEEE},
  address = {Kyoto, Japan},
  doi = {10.1109/IROS47612.2022.9982093}
}
```

## Installation

To install, run `pip install -r requirements.txt`. Tested with Python 3.8.10 on Windows 10 and Ubuntu Linux 20.04.

## Demo

`meta_learning_experiments/experiments/spike/demo.ipynb` contains an interactive demo for optimizing a probe search strategy with ∂PSE on synthetic robot data.