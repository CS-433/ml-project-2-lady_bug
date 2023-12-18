# ML4Science Project

Train PINN for the cosmic epoch of reionization:
- Implement the Hydrogen ODE in a FCNN for EoR simulations
- The ODE for HII fraction: $\frac{dx}{dt} = (1 - x)(\Gamma + n_e C_H) - xn_e\alpha_H$

#### Deliverables
- Written report: max 4 pages
- Code: in Pytorch
    * Results reproducibility
    * External libraries citations

#### Timeline
Week 1: (16 Nov)
- Intro and first data analysis

Week 2-4: (23, 30 Nov and 7 Dec)
- Weekly updates presented by one student

Week 5: (14 Dec)
- Finalise the results start writing the report

Week 6: (18 Dec)
- Finalise the report (plots, text, review report, etc.)

## Codebase structure
Notebooks `PINN_for_hydrogen_ionization.ipynb` and `test_simulation.ipynb` contain the experiments with different models for the task of predicting hydrogen fraction. Notebook `example_PINN.ipynb` contains the example of PINN. Code for the models, data loading and training is in the `src` folder.