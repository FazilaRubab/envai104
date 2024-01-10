"""
================================
Experiments
================================
"""


import matplotlib.pyplot as plt
from ai4water.experiments import MLRegressionExperiments
from utils import make_data, set_rcParams, print_version_info
# %%

print_version_info()

# %%

SAVE= False


set_rcParams()

# %%

data, enc = make_data()

print(data.shape)
# %%
# Removal Efficiency
# ======================
output_features= 'removal%'

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


# %%
# Rate Constant
# ======================
output_features= 'K Reaction rate constant (k 10-2min-1)'

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