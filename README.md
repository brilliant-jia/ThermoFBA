# ThermoFBA

ThermoFBA is a thermodynamics-constrained flux balance analysis workflow implemented with Pyomo, CPLEX, and IPOPT.

## Environment

The code was developed and tested in a Conda environment named `FBA` on 64-bit Windows.

| Component | Version |
| --- | --- |
| Python | 3.7.12 |
| NumPy | 1.19.2 |
| SciPy | 1.4.1 |
| pandas | 1.3.5 |
| Matplotlib | 3.5.3 |
| Pyomo | 6.6.2 |
| tqdm | 4.66.1 |
| IBM ILOG CPLEX | 12.10.0.0 |
| IPOPT | 3.11.1 |

CPLEX must be installed and licensed separately. Both the CPLEX Python API and the CPLEX executable must be available in the environment.

## Python Scripts

Run all scripts from the repository root because they use relative paths.

### ThermoFBA

- `solve.py`: the main thermoFBA script. It builds the thermodynamic model, obtains a mixed-integer solution pool with CPLEX, refines the candidate solutions with IPOPT, and saves the results as `output_*.pkl`.
- `model.py`: constructs the Pyomo mixed-integer and nonlinear thermodynamic models.
- `cplex_solu.py`: runs CPLEX and converts the solution pool into Python objects.
- `nlpsolver.py`: configures IPOPT for nonlinear refinement.

Run thermoFBA with:

```powershell
conda activate FBA
python solve.py
```

### Data Processing and Plotting

- `custom_yield_rate.py`: processes thermoFBA result files, calculates growth yields, exports CSV data, and generates figures.
- `calculate_ag_new.py`: calculates Gibbs-energy-dissipation parameters from `output_*.pkl` files and generates figures.
- `yield_comparison.py`: generates yield and Gibbs-energy comparison figures from the prepared simulation data.
- `yield_comparison_new.py`: provides an alternative yield-processing and plotting workflow.

The data-processing scripts require the corresponding carbon-source result folders and their `data.pkl` and `output_*.pkl` files. `simulation_data_4.0.pkl` contains prepared data used by the yield-comparison scripts.

