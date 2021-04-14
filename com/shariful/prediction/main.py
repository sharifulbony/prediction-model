import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import missingno as msno
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import levene
from scipy.stats import shapiro
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import scale
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve



def process_data():
    filterwarnings("ignore")
    raj_file_path = "../../../data/heart.csv"
    Heart = pd.read_csv(raj_file_path)
    print(Heart.head())
    data = Heart.copy()

    dataV = data.copy()
    dataV["sex"] = pd.Categorical(dataV["sex"])
    dataV["cp"] = pd.Categorical(dataV["cp"])
    dataV["fbs"] = pd.Categorical(dataV["fbs"])
    dataV["restecg"] = pd.Categorical(dataV["restecg"])
    dataV["exng"] = pd.Categorical(dataV["exng"])
    dataV["slp"] = pd.Categorical(dataV["slp"])
    dataV["caa"] = pd.Categorical(dataV["caa"])
    dataV["thall"] = pd.Categorical(dataV["thall"])
    dataV["output"] = pd.Categorical(dataV["output"])

    df = data.select_dtypes(include=["float64", "int64", "int32"])
    print(data.shape)
    print("------" * 20)
    print(data.columns)
    print("------" * 20)
    print(data.info())
    print("------" * 20)
    print(data.describe())
    print("------" * 20)
    print(df.corr())
    print("------" * 20)
    print(data["sex"].value_counts())
    print("------" * 20)
    print(data["cp"].value_counts())
    print("------" * 20)
    print(data["fbs"].value_counts())
    print("------" * 20)
    print(data["restecg"].value_counts())
    print("------" * 20)
    print(data["exng"].value_counts())
    print("------" * 20)
    print(data["slp"].value_counts())
    print("------" * 20)
    print(data["caa"].value_counts())
    print("------" * 20)
    print(data["thall"].value_counts())
    print("------" * 20)
    print(data["output"].value_counts())
    print("------" * 20)
    print(data.groupby(["sex", "output"])["trtbps"].mean())
    print("------" * 20)
    print(data.groupby(["sex", "output"])["chol"].mean())
    print("------" * 20)
    print(data.groupby(["sex", "output"])["thalachh"].mean())
    print("------" * 20)
    print(data.groupby(["sex", "output"])["oldpeak"].mean())
    print("------" * 20)
    print(data.isnull().sum())
    corrPearson = data.corr(method="pearson")
    corrSpearman = data.corr(method="spearman")
    figure = plt.figure(figsize=(10, 8))
    sns.heatmap(corrPearson, annot=True, cmap='RdYlGn', vmin=-1, vmax=+1)
    plt.title("PEARSON")
    plt.xlabel("COLUMNS")
    plt.ylabel("COLUMNS")
    plt.show()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    process_data()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
