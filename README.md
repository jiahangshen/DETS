## 🛠️ Installation

We recommend using Conda to manage the environment to ensure reproducibility.

```bash
# 1. Create a new environment
conda create -n dets python=3.8
conda activate dets

# 2. Install dependencies
pip install -r requirements.txt

```

## 🚀 Usage

The framework is modularized by domain. Below are the instructions for running experiments in each category.

### 1. Materials Science (MatDB)

Located in `MatDB/`, this module handles property prediction for materials using the `bandgap` and `enthalpy` datasets.

```bash
# Example: Training on Bandgap data
python MatDB/icl_dets_train.py 

```

### 2. Drug-Target Affinity (DTA)

Located in `DTA/`, this module predicts the binding affinity between drugs and targets using the `drug-target` dataset.

```bash
# Example: Training on Drug-Target data
python DTA/icl_dets_train.py 

```

### 3. Solute-Solvent Properties (SSEP)

Located in `SSEP/`, this module focuses on solvation energy and other chemical properties using the `solute-solvent` dataset.

```bash
# Example: Solvation energy prediction
python SSEP/icl_dets_train.py 

```
### 4.Molecule Enthalpy

Located in `DETS/`, this module handles property prediction for materials using the `enthalpy` datasets.

```bash
# Example: Training on Molecule Enthalpy data
python DETS/icl_dets_train.py 

```
## 📝 Methodology

DETS introduces a **Student- posterior predictive distribution** derived from a Normal-Inverse-Gamma (NIG) prior. This allows for robust regression even in the presence of heavy-tailed noise.

Key features include:

* **Uncertainty Decomposition:**Analytically separating total variance into data noise and model uncertainty.
* **Interval-Censored Likelihood (ICL):** Maximizing probability mass within tolerance intervals rather than point estimates.
