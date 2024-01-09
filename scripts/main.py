"""
====================
main
====================
"""

from utils import make_data, set_rcParams, regression_plot, get_predictions
from utils import residual_plot, shap_plot, print_version_info
from utils import cumulative_probability_plot

# %%
print_version_info()

# %%
set_rcParams()

# %%
data, enc = make_data()
print(data.shape)

# %%
# Removal Efficiency
# ======================

# %%

rem_true, rem_pred, rem_scores = get_predictions("removal%", cv="KFold", n_splits=5)
rem_pred[rem_pred<0.0] = 0.0 # todo

# %%
# Scatter plot

regression_plot(rem_true, rem_pred, hist_bins=25,  label="Removal %")

# %%
# Residual plot

residual_plot(rem_true, rem_pred, x_axis_label= "Removal %")

# %%
# Error plot

cumulative_probability_plot(rem_true, rem_pred, x_axis_label= "Removal %")

# %%
#SHAP plot
shap_plot(output_features= "removal%")

# %%
# Rate Constant
# ==================
k_true, k_pred, k_scores = get_predictions("K Reaction rate constant (k 10-2min-1)",
                                           cv="KFold",
                                              n_splits=5)
k_pred[k_pred<0.0] = 0.0 # todo

# %%
# Regression plot

regression_plot(k_true, k_pred, hist_bins=25,  label="Rate Constant (k)")


# %%
# Residual plot
residual_plot(k_true, k_pred, x_axis_label= "Rate Constant (k)")

# %%
# Error Plot
cumulative_probability_plot(k_true, k_pred, x_axis_label= "Rate Constant (k)")

# %%
# Shap Plot
shap_plot(output_features= "K Reaction rate constant (k 10-2min-1)")