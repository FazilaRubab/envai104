"""
====================
utils
====================
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from ai4water.utils.utils import get_version_info
from typing import Union, List, Tuple

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
    fpath = '../data/master_sheet_QDS_CeZn.xlsx'
    return pd.read_excel(fpath)

# %%
def get_data(
        input_features:Union[str, List[str]] = None,
        output_features:Union[str, List[str]] = 'removal%'
)->pd.DataFrame:
    def_inputs = [
        'time_min', 'PMS_concentration g/L',
        'Co (intial content of DS pollutant)',
        'IBU_conc_mg/L',
         'catalyst dosage_g/L', 'cycle_no', 'Ct'
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

def make_data(
        input_features=None,
        output_features= ['removal%', 'K Reaction rate constant (k 10-2min-1)'],
        encoding:str = "le",
)->Tuple[pd.DataFrame, dict]:

    data = get_data(input_features, output_features)

    encoders = {}


    if encoding == "ohe":
        # applying One Hot Encoding
        data, _, encoders['ion_type'] = _ohe_column(data, 'ion_type')
        data, _, encoders['ion_type'] = _ohe_column(data, 'Catalyst')

    elif encoding == "le":
        # applying Label Encoding
        data, encoders['ion_type'] = le_column(data, 'ion_type')
        data, encoders['catalyst'] = le_column(data, 'catalyst')

    return data, encoders
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






