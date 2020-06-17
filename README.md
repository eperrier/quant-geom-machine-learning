
# Quantum Geometric Machine Learning for Quantum Circuits and Control

This repository contains the code base for implementation of the quantum geometric machine learning (QGML) methods for and simulations of time-optimal quantum circuit synthesis using discretised approximations to geodesics on certain SU($2^n$) Lie group manifolds of relevance to mult-quibit systems as set-out in the paper <i> Quantum Geometric Machine Learning for Quantum Circuits and Control</i> by Perrier et al. of the Centre for Quantum Software and Information at the University of Technology, Sydney. A preprint of the paper is [available at https://arxiv.org/[*]]. The code is implemented in Python, primarily drawing upon TensorFlow >= 2.2 and Python > 3.7 along with a range of other standard packages, including Qutip >= 4.0. 

The repository is structured as follows:

* <b>simulation.py</b>: this files contains Python code written for generating sequences of unitary propagators along with a range of other supplementary hyperparameters and datasets used as inputs in to the machine learning models detailed in the paper. It represents a Python adaptation of Mathematica code from [1]. The Python code is based on a class construction that outputs a range of such sequences and unitary propagators for use as training, validation and test data with respect to the machine learning models in the paper. It can be called via example Jupyter Notebooks. The code is the means by which to generate training and validation datasets used as the basis for the results in the paper.

* <b>QGML.ipynb</b>: a Jupyter Notebook containing the QGML class which in turn contains each of the greybox models discussed in the paper;

* <b>QGML - original model.ipynb</b>: a Jupyter Notebook containing the an adapted implementation of the original model from [1];

* <b>customlayers.py</b>: a file containing customised layers called by the QGML class;

* <b>holonomy.py</b>: implemtnation in Python code of results from [2] regarding analytic expressions for holonomic (geodesic) paths on SU(2).
    
Each file contains commentary to assist researchers in understanding the architecture and how various coding modules fit together.

<b> Datasets </b>

A number of datasets (generaed using simulation.py called in a Jupyter notebook) are include in compressed .tar format. They are provided for the benefit of researchers and for any attempts at replication of the results of the paper and in the interests of promoting open research and collaboration. The naming convention of these files indicates their hyperparameters (see paper for details), for example the file:

su2_ntrain1000_nseg10_hscale0.1_su2gen1.pickle

indicates a data structure (a Python class structure saved via pickle format) with:
* Lie group SU(2) and Lie algebra su(2);
* ntrain1000: 1000 training examples;
* nseg10: 10 segements in the discretised approximation of the geodesic quantum circuit;
* hscale: the time-step h for which subunitaries are evolved.

All code files should be placed in the same folder. Each file will need to be extracted into the same folder as the code.

[1] M. Swaddle, L. Noakes, H. Smallbone, L. Salter, and J. Wang,Generating three-qubit quantum circuits with neural networks,Physics Letters A381, 3391–3395 (2017).

[2] A. D. Boozer, Time-optimal synthesis of su(2) transformationsfor a spin-1/2 system, Physical Review A85, 012317 (2012).
