# %% [markdown]
# # Desempenho de algoritmos de classificação na detecção de intrusão em redes de dispositivos IoT

# %% [markdown]
# ## Importação de bibliotecas
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from dtos import ModelDataset
from utils import correlation, evaluate_model, generate_dataframe, save_results

# %% [markdown]
# ## Constantes
CURRENT_DIR = Path.cwd()
PROJECT_DIR = CURRENT_DIR.parent if CURRENT_DIR.name == "src" else CURRENT_DIR
DATASET_DIRECTORY = PROJECT_DIR / "CICIoT2023"
DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"
DEV = "dev"
PROD = "five_percent"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
# %% [markdown]
# ## Parâmetros
GENERATE_DATASETS = False
SAVE_PLOTS = False

FILES_SUFFIX = DEV
DATASET_FILE = DATA_DIR / f"dataset_{FILES_SUFFIX}.csv"
RESULTS_FILE = PROJECT_DIR / f"results_{FILES_SUFFIX}.csv"

NO_CLASSES = 2
FEATURE_SELECTION = True

LOGISTIC_REGRESSION = True
PERCEPTRON = True
ADABOOST = True
DNN = True
RANDOM_FOREST = True

assert NO_CLASSES in [2, 8, 34]

# %% [markdown]
# ## Criando datasets menores
if GENERATE_DATASETS:
    df_sets = sorted(glob.glob(f"{DATASET_DIRECTORY}/*.csv"))

# %% [markdown]
# ### Dataset de desenvolvimento (0.5%)
if GENERATE_DATASETS:
    generate_dataframe(file_list=df_sets, percentage=0.5).write_csv(
        file=DATA_DIR / f"dataset_{DEV}.csv"
    )

# %% [markdown]
# ### Dataset final (5%)
if GENERATE_DATASETS:
    generate_dataframe(file_list=df_sets, percentage=5).write_csv(
        file=DATA_DIR / f"dataset_{PROD}.csv"
    )

# %% [markdown]
# ## Pré-processamento
# ### Lendo o dataset
df = pl.read_csv(DATASET_FILE)
training_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
del df

# %% [markdown]
# ### Separando Features e Label
X_columns = training_data.columns[:-1]
y_column = "label"

# %% [markdown]
# ### 8 Classes
if NO_CLASSES == 8:
    eight_classes = {}

    eight_classes["BenignTraffic"] = "Benign"

    eight_classes["DDoS-RSTFINFlood"] = "DDoS"
    eight_classes["DDoS-PSHACK_Flood"] = "DDoS"
    eight_classes["DDoS-SYN_Flood"] = "DDoS"
    eight_classes["DDoS-UDP_Flood"] = "DDoS"
    eight_classes["DDoS-TCP_Flood"] = "DDoS"
    eight_classes["DDoS-ICMP_Flood"] = "DDoS"
    eight_classes["DDoS-SynonymousIP_Flood"] = "DDoS"
    eight_classes["DDoS-ACK_Fragmentation"] = "DDoS"
    eight_classes["DDoS-UDP_Fragmentation"] = "DDoS"
    eight_classes["DDoS-ICMP_Fragmentation"] = "DDoS"
    eight_classes["DDoS-SlowLoris"] = "DDoS"
    eight_classes["DDoS-HTTP_Flood"] = "DDoS"

    eight_classes["DoS-UDP_Flood"] = "DoS"
    eight_classes["DoS-SYN_Flood"] = "DoS"
    eight_classes["DoS-TCP_Flood"] = "DoS"
    eight_classes["DoS-HTTP_Flood"] = "DoS"

    eight_classes["Mirai-greeth_flood"] = "Mirai"
    eight_classes["Mirai-greip_flood"] = "Mirai"
    eight_classes["Mirai-udpplain"] = "Mirai"

    eight_classes["Recon-PingSweep"] = "Recon"
    eight_classes["Recon-OSScan"] = "Recon"
    eight_classes["Recon-PortScan"] = "Recon"
    eight_classes["VulnerabilityScan"] = "Recon"
    eight_classes["Recon-HostDiscovery"] = "Recon"

    eight_classes["DNS_Spoofing"] = "Spoofing"
    eight_classes["MITM-ArpSpoofing"] = "Spoofing"

    eight_classes["BrowserHijacking"] = "Web"
    eight_classes["Backdoor_Malware"] = "Web"
    eight_classes["XSS"] = "Web"
    eight_classes["Uploading_Attack"] = "Web"
    eight_classes["SqlInjection"] = "Web"
    eight_classes["CommandInjection"] = "Web"

    eight_classes["DictionaryBruteForce"] = "BruteForce"

    training_data = training_data.with_columns(
        pl.col(y_column).replace(eight_classes, default=-1)
    )
    test_data = test_data.with_columns(
        pl.col(y_column).replace(eight_classes, default=-1)
    )
# %% [markdown]
# ### 2 Classes
if NO_CLASSES == 2:
    two_classes = {}

    two_classes["BenignTraffic"] = "Benign"

    two_classes["DDoS-RSTFINFlood"] = "Attack"
    two_classes["DDoS-PSHACK_Flood"] = "Attack"
    two_classes["DDoS-SYN_Flood"] = "Attack"
    two_classes["DDoS-UDP_Flood"] = "Attack"
    two_classes["DDoS-TCP_Flood"] = "Attack"
    two_classes["DDoS-ICMP_Flood"] = "Attack"
    two_classes["DDoS-SynonymousIP_Flood"] = "Attack"
    two_classes["DDoS-ACK_Fragmentation"] = "Attack"
    two_classes["DDoS-UDP_Fragmentation"] = "Attack"
    two_classes["DDoS-ICMP_Fragmentation"] = "Attack"
    two_classes["DDoS-SlowLoris"] = "Attack"
    two_classes["DDoS-HTTP_Flood"] = "Attack"

    two_classes["DoS-UDP_Flood"] = "Attack"
    two_classes["DoS-SYN_Flood"] = "Attack"
    two_classes["DoS-TCP_Flood"] = "Attack"
    two_classes["DoS-HTTP_Flood"] = "Attack"

    two_classes["Mirai-greeth_flood"] = "Attack"
    two_classes["Mirai-greip_flood"] = "Attack"
    two_classes["Mirai-udpplain"] = "Attack"

    two_classes["Recon-PingSweep"] = "Attack"
    two_classes["Recon-OSScan"] = "Attack"
    two_classes["Recon-PortScan"] = "Attack"
    two_classes["VulnerabilityScan"] = "Attack"
    two_classes["Recon-HostDiscovery"] = "Attack"

    two_classes["DNS_Spoofing"] = "Attack"
    two_classes["MITM-ArpSpoofing"] = "Attack"

    two_classes["BrowserHijacking"] = "Attack"
    two_classes["Backdoor_Malware"] = "Attack"
    two_classes["XSS"] = "Attack"
    two_classes["Uploading_Attack"] = "Attack"
    two_classes["SqlInjection"] = "Attack"
    two_classes["CommandInjection"] = "Attack"

    two_classes["DictionaryBruteForce"] = "Attack"

    training_data = training_data.with_columns(
        pl.col(y_column).replace(two_classes, default=-1)
    )
    test_data = test_data.with_columns(
        pl.col(y_column).replace(two_classes, default=-1)
    )

# %% [markdown]
# ### Separando as colunas
X_train = training_data.select(X_columns)
y_train = training_data.select(y_column)
X_test = test_data.select(X_columns)
y_test = test_data.select(y_column).to_numpy()

# %% [markdown]
# ### Desvio Padrão
std_devs = X_train.std()
cols_to_keep = [col for col in X_train.columns if std_devs[col][0] > 0.0]
X_train_filtered = X_train.select(cols_to_keep)
X_test_filtered = X_test.select(cols_to_keep)

# %% [markdown]
# ### Correlação
sns.set_theme(rc={"figure.figsize": (75, 25)})

corr = X_train_filtered.corr()
correlation_plot = sns.heatmap(
    data=corr,
    cbar=False,
    cmap="coolwarm",
    fmt=".3f",
    mask=np.triu(corr),
    annot=True,
    xticklabels=corr.columns,
    yticklabels=corr.columns,
)
if SAVE_PLOTS:
    correlation_plot.get_figure().savefig(FIGURES_DIR / "correlation_plot.png")

# %%
if FEATURE_SELECTION:
    correlated_columns = correlation(X_train_filtered, 0.9)

    X_train = X_train_filtered.drop(correlated_columns)
    X_test = X_test_filtered.drop(correlated_columns)

# %% [markdown]
# ### Balanceamento com SMOTE
print("Before")
print(y_train.group_by("label").len())

smote = SMOTE(random_state=42)
pd_X_resampled, pd_y_resampled = smote.fit_resample(
    X_train.to_pandas(), y_train.to_pandas()
)
X_resampled, y_resampled = pl.from_pandas(pd_X_resampled), pl.from_pandas(
    pd_y_resampled
)

print("After")
print(y_resampled.group_by("label").len())

# %% [markdown]
# ### Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## ML Models
data = ModelDataset(
    x_train=X_train_scaled,
    y_train=y_resampled.to_numpy(),
    x_test=X_test_scaled,
    y_test=y_test,
)

# %% [markdown]
# ### Regressão Logística
if LOGISTIC_REGRESSION:
    logistic_regression_results = evaluate_model(
        model=LogisticRegression(n_jobs=-1, max_iter=10000),
        data=data,
        feature_selection_bool=FEATURE_SELECTION,
    )
    save_results(results=logistic_regression_results, save_file=RESULTS_FILE)

# %% [markdown]
# ### Perceptron de uma camada
if PERCEPTRON:
    perceptron_results = evaluate_model(
        model=Perceptron(), data=data, feature_selection_bool=FEATURE_SELECTION
    )
    save_results(results=perceptron_results, save_file=RESULTS_FILE)

# %% [markdown]
# ### AdaBoost
if ADABOOST:
    adaboost_results = evaluate_model(
        model=AdaBoostClassifier(),
        data=data,
        feature_selection_bool=FEATURE_SELECTION,
    )
    save_results(results=adaboost_results, save_file=RESULTS_FILE)

# %% [markdown]
# ### DNN (Perceptron multicamadas)
if DNN:
    dnn_results = evaluate_model(
        model=MLPClassifier(
            hidden_layer_sizes=(100,),
            activation="relu",
            solver="adam",
            random_state=42,
        ),
        data=data,
        feature_selection_bool=FEATURE_SELECTION,
    )
    save_results(results=dnn_results, save_file=RESULTS_FILE)

# %% [markdown]
# ### Floresta Aleatória
if RANDOM_FOREST:
    random_forest_results = evaluate_model(
        model=RandomForestClassifier(),
        data=data,
        feature_selection_bool=FEATURE_SELECTION,
    )
    save_results(results=random_forest_results, save_file=RESULTS_FILE)
