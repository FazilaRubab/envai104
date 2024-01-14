"""
====================
utils
====================
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from ai4water.utils.utils import get_version_info, METRIC_TYPES, TrainTestSplit
from sklearn.model_selection import KFold, GroupKFold
from typing import Union, List, Tuple, Callable
from SeqMetrics import RegressionMetrics
from easy_mpl import regplot
import shap
from shap import summary_plot
from ai4water import Model

# %%
SAVE= False

# %%

LABEL_MAP = {
    'time_min': 'Time (min)',
    'PMS_concentration g/L': "PMS Concentration (g/L)",
    'Co (intial content of DS pollutant)': 'Concentration (intial content of DS pollutant)',
    'MO_conc_mg/L': "MO Concentration (mg/L)",
    'NP_conc_mg/L': "NP Concentration (mg/L)",
    'NX_conc_mg/L': "NX Concentration (mg/L)",
    'TC_conc_mg/L': "TC Concentration (mg/L)",
    'IBU_conc_mg/L': "IBU Concentration (mg/L)",
    'catalyst dosage_g/L': "Catalyst Dosage (g/L)",
    'pH': "pH",
    'system': 'System',
    'PS g/L': 'PS g/L',
    'H2O2 micro-L': 'H2O2 micro-L',
    'Light': 'Light',
    'Sonication': 'Sonication',
    'ion_conc.': 'Ion Concentration',
    'ion_type': 'Ion Type',
    'O2_quenching': 'O2 Quenching',
    'h+_quenching': 'h+ Quenching',
    'OH_quenching': 'OH Quenching',
    'so4_quenching': 'SO4 Quenching',
    'cycle_no': 'Cycle No',
    'Ct': 'Ct',
    'qe': 'qe',
    'removal%': 'Removal %',
    "K Reaction rate constant (k 10-2min-1)": "Rate Constant (k)"
}

# %%
def load_data() -> pd.DataFrame:
    fpath = '../data/master_sheet.xlsx'
    return pd.read_excel(fpath)

# %%
def get_data(
        input_features:Union[str, List[str]] = None,
        output_features:Union[str, List[str]] = 'removal%'
)->pd.DataFrame:
    def_inputs = [
        'PMS_concentration g/L', 'Co (intial content of DS pollutant)',
        'MO_conc_mg/L', 'NP_conc_mg/L', 'NX_conc_mg/L', 'ion_type',
        'TC_conc_mg/L', 'IBU_conc_mg/L', 'catalyst dosage_g/L', 'pH', 'PS g/L', 'H2O2 micro-L',
        'Light', 'Sonication', 'O2_quenching', 'h+_quenching', 'OH_quenching', 'so4_quenching',
        'cycle_no', 'system', 'Ct', 'time_min'
    ]

    if input_features is None:
        input_features = def_inputs
    elif isinstance(input_features, str):
        input_features = [input_features]
        assert all(feature in def_inputs for feature in input_features)
    else:
        raise ValueError

    if not isinstance(output_features, list):
        output_features = [output_features]

    for feature in output_features:
        assert feature in ['removal%', 'K Reaction rate constant (k 10-2min-1)']

    df = load_data()
    df = df[input_features + output_features]
    return df

# %%
def _ohe_column(df:pd.DataFrame, col_name:str)->tuple:
    # function for OHE
    assert isinstance(col_name, str)

    if col_name in df:
        # setting sparse to True will return a scipy.sparse.csr.csr_matrix
        # not a numpy array
        encoder = OneHotEncoder(sparse=False)
        ohe_cat = encoder.fit_transform(df[col_name].values.reshape(-1, 1))
        cols_added = [f"{col_name}_{i}" for i in range(ohe_cat.shape[-1])]

        df[cols_added] = ohe_cat

        df.pop(col_name)

        return df, cols_added, encoder
    return df, None, None

# %%
def print_version_info():
    info = get_version_info()

    for k,v in info.items():
        print(k, v)
    return

def le_column(df:pd.DataFrame, col_name)->tuple:
    """label encode a column in dataframe"""
    if col_name in df:
        encoder = LabelEncoder()
        df[col_name] = encoder.fit_transform(df[col_name])

        return df, encoder
    return df, None
# %%
def set_rcParams(**kwargs):
    plt.rcParams.update({'axes.labelsize': '14'})
    plt.rcParams.update({'axes.labelweight': 'bold'})
    plt.rcParams.update({'xtick.labelsize': '12'})
    plt.rcParams.update({'ytick.labelsize': '12'})
    plt.rcParams.update({'legend.title_fontsize': '12'})
    plt.rcParams.update({"font.family": "Times New Roman"})

    for k,v in kwargs.items():
        plt.rcParams[k] = v
    return
# %%
def read_data(
        inputs=None,
        outputs="removal%",
        encoding="le"
):
    df = pd.read_excel("../data/master_sheet.xlsx")


    default_inputs = [
        'PMS_concentration g/L', 'Co (intial content of DS pollutant)',
        'MO_conc_mg/L', 'NP_conc_mg/L', 'NX_conc_mg/L', 'ion_type',
        'TC_conc_mg/L', 'IBU_conc_mg/L', 'catalyst dosage_g/L', 'pH', 'PS g/L', 'H2O2 micro-L',
        'Light', 'Sonication', 'O2_quenching', 'h+_quenching', 'OH_quenching', 'so4_quenching',
        'cycle_no', 'system', 'Ct', 'time_min'
    ]

    if inputs is None:
        inputs = default_inputs

    if outputs is None:
        outputs = ['removal%', "K Reaction rate constant (k 10-2min-1)"]
    elif not isinstance(outputs, list):
           outputs = [outputs]

    df = df[inputs + outputs]

    ads_encoder = None


    if encoding=="ohe":
        # applying One Hot Encoding
        df, _, ads_encoder = _ohe_column(df, 'system')

    elif encoding == "le":
        # applying Label Encoding
        df, ads_encoder = le_column(df, 'system')

    return df.dropna(), ads_encoder

# %%
def prepare_data(
        inputs=None,
        outputs="removal%",
        encoding="le"
):
    data, encoder = read_data(
        inputs=inputs,
        outputs=outputs,
        encoding=encoding
    )

    input_features = data.iloc[:, :-1]  # Select all rows and all columns except the last one
    output_features = data.iloc[:, -1:]  # Select all rows and only the last column

    train_idx, test_idx = TrainTestSplit(seed=313).split_indices(data)

    train_x = input_features.iloc[train_idx]
    test_x = input_features.iloc[test_idx]
    train_y = output_features.iloc[train_idx]
    test_y = output_features.iloc[test_idx]

    return train_x, test_x, train_y, test_y, input_features, output_features

# %%
def get_predictions(
        output,
        cv = None,
        n_splits=5,
):
    """
    Trains the models for predicting of the specific output
    """
    model = Model(model='CatBoostRegressor', verbosity=-1)



    if cv:

        assert cv in ['KFold', 'GroupKFold']
        all_data, _ = read_data(outputs=output)

        x = all_data.iloc[:, 0:-1]
        y = all_data.iloc[:, -1]

        cv_args = {'n_splits': n_splits}
        if cv == "GroupKFold":
            cv_args.update({"groups": x['Adsorbent'].values.astype(int).reshape(-1,)})


        cv_scores, observations, prediction, indices = cross_val_score(
            model, x, y,
            cross_validator=cv,
            cross_validator_args=cv_args,
            scoring=['r2', 'r2_score', 'rmse', 'mae'])

        prediction = np.concatenate(prediction)
        observations = np.concatenate(observations)
        indices = np.concatenate(indices)
        performance = np.array(cv_scores)
        df = pd.DataFrame(
            np.vstack([prediction, observations]).transpose(),
            index=indices, columns=['prediction', 'observation']
        ).sort_index()
        obs = df['observation'].values.reshape(-1,)
        pred = df['prediction'].values.reshape(-1,)
    else:
        TrainX, TestX, TrainY, TestY, _, _ = prepare_data(
            outputs=output)
        performance = model.fit(x=TrainX, y=TrainY.values)

        all_data, _ = read_data(outputs=output)
        obs = all_data.iloc[:, -1].values
        pred = model.predict(all_data.iloc[:, 0:-1])

    return obs, pred, performance

def regression_plot(true, prediction, hist_bins=20, label="Removal Efficiency %", show=True):

    RIDGE_LINE_KWS = [{'color': '#003B46', 'lw': 1.0}, {'color': '#2E8B57', 'lw': 1.0}]
    HIST_KWS = [{'color': '#41BA90', 'bins': hist_bins}, {'color': '#3AB763', 'bins': hist_bins}]

    ax = regplot(true, prediction, marker_color='#036666', marker_size=35,
                 scatter_kws={'marker': "D", 'edgecolors': '#88D4AB'},
                 line_color='black', marginals=True, show=False, ridge_line_kws=RIDGE_LINE_KWS,
                 hist=True,
                 hist_kws=HIST_KWS, ax_kws=dict(xlabel=f"Experimental {label}",
                                                ylabel=f'Predicted {label}'))

    # Increase font size of tick labels
    ax.tick_params(axis='both', which='major', labelsize=16)

    metrics = RegressionMetrics(true, prediction)

    r2 = metrics.r2()
    rmse = metrics.rmse()

    ax.annotate(f'$R^2$= {round(r2, 2)}',
                xy=(0.95, 0.30),
                xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='top',
                fontsize=12, weight="bold")
    ax.annotate(f'RMSE= {round(rmse, 2)}',
                xy=(0.95, 0.20),
                xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='top',
                fontsize=12, weight="bold")

    if SAVE:
        plt.savefig(f"results/figures/regression_{label}.png", dpi=600, bbox_inches="tight")

    if show:
        plt.show()
    return ax


# %%
def cross_val_score(
        model,
        x,
        y,
        scoring: Union [str, List[str], Callable] = 'r2',
        cross_validator = "KFold",
        cross_validator_args = None,
        process_results:bool = False,
        verbosity=1,
) -> Tuple[List, List, List, List]:
    """
    computes cross validation score

    Parameters
    ----------
        x :
            input data
        y :
            output corresponding to ``x``.
        scoring : (default=None)
            performance metric to use for cross validation.
            If None, it will be taken from config['val_metric']
            If callable then it must be a function which can take true and predicted
            arrays and return a float.

        process_results : bool, optional
            whether to process results at each cv iteration or not
        cross_validator:

        cross_validator_args:

        verbosity:

        model:

    Returns
    -------
    list
        cross validation score for each of metric in scoring

    Example
    -------
    >>> from ai4water.datasets import busan_beach
    >>> from ai4water import Model
    >>> model = Model(model="RandomForestRegressor",
    >>>               cross_validator={"KFold": {"n_splits": 5}})
    >>> model.cross_val_score(data=busan_beach())

    We can also have our own performance metric as scoring

    >>> from ai4water.datasets import MtropicsLaos
    >>> df = MtropicsLaos().make_classification(lookback_steps=1)
    >>> def f1_score_(t,p)->float:
    >>>    return ClassificationMetrics(t, p).f1_score(average="macro")
    >>> model = Model(model="RandomForestClassifier",
    ... cross_validator={"KFold": {"n_splits": 5}},)
    >>> model.cross_val_score(data=df, scoring=f1_score_)

    Note
    ----
        Currently not working for deep learning models.

    """

    if not isinstance(scoring, list):
        scoring = [scoring]

    scores = []
    predictions = []
    observations = []
    test_indices = []

    if cross_validator_args is None:
        cross_validator_args = {'n_splits': 5}

    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values.reshape(-1,)

    if cross_validator == "KFold":
        kf = KFold(n_splits=cross_validator_args['n_splits'],
                   shuffle=True,
                   random_state=313)
        spliter = kf.split(x)
    else:
        kf = GroupKFold(n_splits=cross_validator_args['n_splits'])
        spliter = kf.split(x, groups=cross_validator_args['groups'])

    for fold, (tr_idx, test_idx) in enumerate(spliter):
        train_x = x[tr_idx]
        test_x = x[test_idx]
        train_y = y[tr_idx]
        test_y = y[test_idx]

        # make a new classifier/regressor at every fold
        model.build(model._get_dummy_input_shape())

        model.fit(x=train_x, y=train_y.reshape(-1, ))

        # since we have access to true y, it is better to provide it
        # it will be used for processing of results
        pred = model.predict(x=test_x, y=test_y, process_results=process_results)
        predictions.append(pred)
        observations.append(test_y)
        test_indices.append(test_idx)

        metrics = RegressionMetrics(test_y.reshape(-1, 1), pred)

        val_scores = []
        for score in scoring:
            if callable(score):
                val_scores.append(score(test_y.reshape(-1, 1), pred))
            else:
                val_scores.append(getattr(metrics, score)())

        scores.append(val_scores)

        if verbosity > 0:
            print(f'fold: {fold} val_score: {val_scores}')

    return scores, observations, predictions, test_indices


# %%
def fill_val(metric_name, default="min", default_min=99999999):
    if METRIC_TYPES.get(metric_name, default) == "min":
        return default_min
    return 0.0

# %%
def residual_plot(obs, pred ,x_axis_label= "Removal %"):

    # Calculate errors and prepare data for plotting
    errors = obs - pred
    df_er = pd.DataFrame({'Error': errors, 'Prediction': pred, 'Observation': obs})

    # Define colors for the scatter plot and marginals
    scatter_color = '#E88267'
    marginal_color = '#E07A5F'

    # Plot the error plot

    g = sns.jointplot(
        data=df_er,
        x="Prediction",
        y="Error",
        kind="scatter",
        color=scatter_color,  # Specify the scatter plot color here
        marginal_kws=dict(bins=25, color=marginal_color)  # Specify the marginal color here
    )

    ax = g.ax_joint
    ax.axhline(0.0, color='firebrick', linestyle='--', linewidth=2)
    ax.set_ylabel(ylabel='Residuals', fontsize=16, weight='bold')
    ax.set_xlabel(xlabel= f"Predicted {x_axis_label}", fontsize=16, weight='bold')
    ax.set_xticklabels(ax.get_xticks().astype(int), size=14)
    ax.set_yticklabels(ax.get_yticks(), size=14)
    plt.tight_layout()
    if SAVE:
        plt.savefig(f"results/figures/residual_summary_{x_axis_label}.png", dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

# %%
def cumulative_probability_plot(
        obs, pred, x_axis_label="Removal %", figsize=None):

    # Calculate errors
    errors = np.abs(obs - pred)

    # Create a figure and axes for plotting
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the EDF for predictions on the top horizontal axis
    sns.ecdfplot(data=pred, label=x_axis_label, ax=ax, color="teal", linewidth=2.5)

    # Create a twin axes for the bottom horizontal axis (absolute error)
    ax2 = ax.twiny()

    # Plot the EDF for absolute errors on the bottom horizontal axis
    sns.ecdfplot(data=errors, label='Absolute Error', ax=ax2, color="coral", linewidth=2.5)

    # Set labels, ticks, and legends
    ax.set_ylabel('Cumulative Probability', fontsize=18, weight='bold')
    ax.set_xlabel(f"Predicted {x_axis_label}", fontsize=18, weight='bold', color='teal')
    ax2.set_xlabel('Absolute Error', fontsize=18, weight='bold', color="coral")

    # Set ticks with 0.5 intervals for the Absolute Error axis
    min_val = 0  # assuming minimum is 0
    max_val = np.ceil(np.max(errors))  # round up to the next integer
    ticks = np.arange(min_val, max_val + 0.5, 1.5)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"{tick:.1f}" for tick in ticks], size=12)

    # Add grid lines
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7, color='gray')
    ax2.grid(True, which='both', axis='x', linestyle='--', linewidth=0.7, color='gray')

    plt.tight_layout()
    if SAVE:
        plt.savefig(f"results/figures/ERDF_summary_{x_axis_label}.png", dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


# %%
def shap_plot(output_features):
    set_rcParams()

    data, enc = read_data(outputs= output_features)
    print(data.shape)

    input_features = [
        'PMS_concentration g/L', 'Co (intial content of DS pollutant)',
        'MO_conc_mg/L', 'NP_conc_mg/L', 'NX_conc_mg/L', 'ion_type',
        'TC_conc_mg/L', 'IBU_conc_mg/L', 'catalyst dosage_g/L', 'pH', 'PS g/L', 'H2O2 micro-L',
        'Light', 'Sonication', 'O2_quenching', 'h+_quenching', 'OH_quenching', 'so4_quenching',
        'cycle_no', 'system', 'Ct', 'time_min'
    ]

    TrainX, TestX, TrainY, TestY = TrainTestSplit(seed=313).split_by_random(
        data[input_features],
        data[output_features]
    )

    print(TrainX.shape, TrainY.shape, TestX.shape, TestY.shape)
    model = Model(model='CatBoostRegressor', verbosity=-1)


    model.fit(TrainX, TrainY.values)
    train_p = model.predict(TrainX, process_results=False)
    test_p = model.predict(TestX, process_results=False)

    print(train_p.mean())
    print(model._model.feature_importances_)

    exp = shap.TreeExplainer(model=model._model)

    print(exp.expected_value)
    shap_values = exp.shap_values(TrainX, TrainY)

    summary_plot(
        shap_values,
        TrainX,
        max_display=34,
        feature_names=[LABEL_MAP[n] if n in LABEL_MAP else n for n in input_features],
        show=False
    )
    ax = plt.gca()
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_xlabel(f'SHAP value (impact on {LABEL_MAP.get(output_features, output_features)})',
                  fontsize=13, weight="bold")
    ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    plt.tight_layout()
    if SAVE:
        plt.savefig(f"results/figures/shap_summary_{output_features}.png",
                    dpi=600, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    set_rcParams()
    return








