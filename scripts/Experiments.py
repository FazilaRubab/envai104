"""
================================
Experiments
================================
"""

import matplotlib.pyplot as plt
from ai4water.experiments import MLRegressionExperiments
from utils import read_data, set_rcParams, print_version_info
import numpy as np
import random
# %%

print_version_info()

# %%
random.seed(313)
np.random.seed(313)
SAVE= False


set_rcParams()

# %%

# %%
# Removal Efficiency
# ======================

data, enc = read_data()

print(data.shape)
output_features= 'removal%'

# %%
experiments = MLRegressionExperiments(
    input_features = data.columns.tolist()[0:-1],
    output_features = output_features,
    train_fraction=1.0,
    cross_validator={"KFold": {"n_splits": 5}},
    show=False,
    verbosity= -1
)

# %%
experiments.fitcv(
    data=data,
    exclude=[
        "SGDRegressor", 'Lars',
        'LarsCV', 'RANSACRegressor',
        'OneClassSVM',
        'GaussianProcessRegressor',
        'DummyRegressor',
        'LassoLarsCV', 'TheilSenRegressor', 'RadiusNeighborsRegressor'
    ],
)


ax = experiments.plot_cv_scores(fill_color="#2596be", patch_artist=True,
                                exclude="LinearSVR", figsize=(9, 6))
ax.grid()
if SAVE:
    plt.savefig("results/figures/exp_eff.png", bbox_inches="tight", dpi=600)
plt.tight_layout()
plt.show()


# %%
# Rate Constant
# ======================
output_features = 'K Reaction rate constant (k 10-2min-1)'
data, enc = read_data(outputs= output_features)
# %%
data.shape

# %%
experiments = MLRegressionExperiments(
    input_features = data.columns.tolist()[0:-1],
    output_features = output_features,
    train_fraction=1.0,
    cross_validator={"KFold": {"n_splits": 5}},
    show=False
)

# %%
experiments.fitcv(
    data=data,
    exclude=[
        "SGDRegressor", 'Lars',
        'LarsCV', 'RANSACRegressor',
        'OneClassSVM',
        'GaussianProcessRegressor',
        'DummyRegressor',
        'LassoLarsCV', 'TheilSenRegressor', 'RadiusNeighborsRegressor'
    ],
)


ax = experiments.plot_cv_scores(fill_color="#2596be", patch_artist=True,
                                exclude="LinearSVR", figsize=(9, 6))
ax.grid()
if SAVE:
    plt.savefig("results/figures/exp_eff.png", bbox_inches="tight", dpi=600)
plt.tight_layout()
plt.show()