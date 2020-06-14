
<script type="text/javascript" async
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js? 
config=TeX-MML-AM_CHTML"
</script>

# quant-geom-machine-learning

This repository contains the code base for implementation of the quantum geometric machine learning (QGML) methods for and simulations of time-optimal quantum circuit synthesis using discretised approximations to geodesics on certain SU($2^n$) Lie group manifolds of relevance to mult-quibit systems as set-out in the paper "[Quantum Geometric Machine Learning for Quantum Circuits and Control]" [available at https://arxiv.org/[*]]. The code is implemented in Python, primarily drawing upon TensorFlow >= 2.2 and Python > 3.7 along with a range of other standard packages, including Qutip >= 4.0. 

The repository is structured as follows:

* <b>Simulation</b>: this folder contains Python code "simulation.py" written for generating sequences of unitary propagatorsalong with a range of other supplementary hyperparameters and datasets used as inputs in to the machine learning models detailed in the paper. The code is based on a class construction that outputs a range of such sequences and unitary propagators for use as training, validation and test data with respect to the machine learning models in the paper. It can be called via example Jupyter Notebooks. The subfolder Simulation also contains a range of training and validation datasets used as the basis for the results in the paper - they are provided for the benefit of researchers and for any attempts at replication of the results of the paper.

* <b>Models</b>: this folder contains a range of files for use in replicating the machine learning models discussed in the paper, including:
    * <b>QGML_models.ipynh</b>: a Jupyter Notebook containing the QGML class which in turn contains each of the greybox models discussed in the paper;
    * <b>custom_layers.py</b>: a file containing customised layers called by the QGML class;
    * <b>original_models.ipynb</b>: an implementation of the Mathematica code from [Swaddle (2017)] in Python.
    
Each file contains commentary to assist researchers in understanding the architecture and how various coding modules fit together.
    
