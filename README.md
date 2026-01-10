# Stochastic synaptic dynamics under learning.

## Short Description

This repository contains the scripts that were used to produce the figures of

Stubenrauch, Auer, Kempter, and Lindner, Stochastic synaptic dynamics under learning, 

which is currently under review. For a preprint see
https://arxiv.org/abs/2508.13846

## License

This code is supplied as free software under the Affero GNU General Public License v3 (or later). You can use and modify it, but must cite our work. For details, see the LICENSE.txt file.

## Contributors

- Jakob Stubenrauch (1,2), contact: jakob.stubenrauch at rwth-aachen.de
- Naomi Auer (3)
- Richard Kempter (1,3,4)
- Benjamin Lindner (1,2)

Affiliations:

1. Bernstein Center for Computational Neuroscience Berlin, Philippstrasse 13, Haus 2, 10115 Berlin, Germany
2. Physics Department of Humboldt University Berlin, Newtonstrasse 15, 12489 Berlin, Germany
3. Institute for Theoretical Biology, Department of Biology, Humboldt University Berlin, 10115 Berlin, German
4. Einstein Center for Neurosciences Berlin, Berlin 10117, Germany

## Installation

1. Clone or download this repository
2. Install dependencies and compile Cython extensions:

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Reproducing Figures

To reproduce Figures 2, 4, and 5, run the respective scripts from the synaptic_dynamics directory, e.g.,

```bash
python scripts/fig2.py
```

To reproduce Figures 3, 6, 7, and 8, uncomment the respective function call at the bottom of scripts/other_figs.py and run

```bash
python scripts/other_figs.py
```

Note that evaluating the theoretical results can take several minutes for more involved figures. Performing the numerical simulations can take hundreds of hours, these simulations should be executed on clusters with large numbers of CPUs.