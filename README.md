# ML4Science Project

Train PINN for the cosmic epoch of reionization:
- Implement the Hydrogen ODE in a FCNN for EoR simulations
- The ODE for HII fraction: $\frac{dx}{dt} = (1 - x)\Gamma - n_{gas}\alpha x^2$

The idea of PINN is to use loss function that forces the model to fit a physics law describing the process, for that function, refer to `src/physics_loss.py`.

## Codebase structure
Notebooks `PINN_for_hydrogen_ionization.ipynb` and `test_simulation.ipynb` contain the experiments with different models for the task of predicting hydrogen fraction. Notebook `example_PINN.ipynb` contains the example of PINN for the harmonic oscillator problem. 

Code for the models, data loading and training is in the `src` folder.
| file              |                                                      functionalty |
|-------------------|-------------------------------------------------------------------|
|`constants.py`     | constants used for scaling physical parameters                    |
|`data_handler.py`  | code for data loading for both data and physics loss              |
|`experiment_run.py`| classes for running the experiments and logging with `Tensorboard`|
|`models.py`        | models' classes                                                   |
|`physics_loss.py`  | physics loss functions                                            |
|`utils.py`         | helpful functions to plot the results                             |

Folder `runs` is for the `Tensorboard` logs.

Folder `weekly_meetings` contains presentations form the weekly meetings with our tutor.

Folder `examples` is for the notebooks that were created during the project but were not the final version.


#### Deliverables
- Written report: max 4 pages
- Code: in Pytorch
    * Results reproducibility (In our notebooks the seed is fixed, so the notebooks provide the same results every run.)
    * External libraries citations:
       - **Pytorch**: Paszke, A. et al., 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32. Curran Associates, Inc., pp. 8024–8035. Available at: http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf.
       - **NumPy**: Harris, C.R. et al., 2020. Array programming with NumPy. Nature, 585, pp.357–362.
       - **Pandas**: McKinney, W. & others, 2010. Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference. pp. 51–56.
       - **Pyc2ray**: Hirling P. et al.,2023. pyC^2Ray: A flexible and GPU-accelerated Radiative Transfer Framework for Simulating the Cosmic Epoch of Reionization

#### Timeline
Week 1: (16 Nov)
- Intro and first data analysis

Week 2-4: (23, 30 Nov and 7 Dec)
- Weekly updates presented by one student

Week 5: (14 Dec)
- Finalise the results start writing the report

Week 6: (18 Dec)
- Finalise the report (plots, text, review report, etc.)
