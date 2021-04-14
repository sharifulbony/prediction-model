import pandas as pd
import numpy as np
import seaborn as sns
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

import os
import pickle
from sklearn import preprocessing
from com.shariful.prediction.scrubbing import scrub


def variance(data):
    print("%.4f - %.4f " % levene(data["age"],
                                  data["sex"],
                                  data["marital_status"],
                                  data["mother_age_when_baby_was_born"],
                                  data["symptoms_pertaining_illness"],
                                  data["diagnosed_for"],
                                  data["disability_status"],
                                  data["wt"],
                                  data["no_of_times_conceived"],
                                  data["smoke"],
                                  data["chew"],
                                  data["alcohol"],
                                  data["sought_medical_care"],
                                  data["outcome_pregnancy"]))


def normality(data):
    for i in data.columns:
        print("-----" * 10)
        print("%.3f - %.3f" % shapiro(data[i]))


def process_data(path):
    filterwarnings("ignore")
    data = pd.read_csv(path, delimiter='|', nrows=1000)
    scrub.scrubData(data)
    print(data.shape)
    print(data.columns)


    DataForA = data.copy()
    clf = LocalOutlierFactor()
    clf.fit_predict(DataForA)
    score = clf.negative_outlier_factor_
    scoresorted = np.sort(score)
    print(scoresorted[0:30])
    point = scoresorted[12]
    print(DataForA[score == point])
    against = DataForA < point
    print(DataForA[against].notna().sum())
    values = DataForA > point
    print(DataForA[values].notna().sum())

    x = data.drop("outcome_pregnancy", axis=1)
    y = data["outcome_pregnancy"]
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.20, random_state=42)

    lm = LinearRegression().fit(xTrain, yTrain)
    Partial_least_squares_regression = PLSRegression().fit(xTrain, yTrain)
    ridge = Ridge().fit(xTrain, yTrain)
    lasso = Lasso().fit(xTrain, yTrain)
    elastic_net = ElasticNet().fit(xTrain, yTrain)
    knn = KNeighborsRegressor().fit(xTrain, yTrain)
    decision_tree = DecisionTreeRegressor(random_state=42).fit(xTrain, yTrain)
    bagging = BaggingRegressor(random_state=42, bootstrap_features=True, verbose=False).fit(xTrain, yTrain)
    random_forest_regressor = RandomForestRegressor(random_state=42, verbose=False).fit(xTrain, yTrain)
    gradient_boosting = GradientBoostingRegressor(verbose=False).fit(xTrain, yTrain)
    xg_boost = XGBRegressor().fit(xTrain, yTrain)
    light_bm_regressor = LGBMRegressor().fit(xTrain, yTrain)
    cat_boost_regressor = CatBoostRegressor(verbose=False).fit(xTrain, yTrain)

    models = [
        lm,
        Partial_least_squares_regression,
        ridge,
        lasso,
        elastic_net,
        knn,
        decision_tree,
        bagging,
        random_forest_regressor,
        gradient_boosting,
        xg_boost,
        light_bm_regressor,
        cat_boost_regressor
    ]

    knn.predict(xTest)

    for model in models:
        name = model.__class__.__name__
        R2CV = cross_val_score(model, xTest, yTest, cv=10, scoring="r2").mean()
        error = -cross_val_score(model, xTest, yTest, cv=10, scoring="neg_mean_squared_error").mean()
        print(name + ": ")
        print("-" * 10)
        print(R2CV)
        print(np.sqrt(error))
        print("-" * 30)
    r = pd.DataFrame(columns=["MODELS", "R2CV"])

    for model in models:
        name = model.__class__.__name__
        R2CV = cross_val_score(model, xTest, yTest, cv=10, scoring="r2").mean()
        result = pd.DataFrame([[name, R2CV * 100]], columns=["MODELS", "R2CV"])
        r = r.append(result)

    figure = plt.figure(figsize=(20, 8))
    sns.barplot(x="R2CV", y="MODELS", data=r, color="k")
    plt.xlabel("R2CV")
    plt.ylabel("MODELS")
    plt.xlim(-50, 100)
    plt.title("MODEL ACCURACY COMPARISON")
    plt.show()

    ols = sm.OLS(yTrain, xTrain).fit()
    print(ols.summary())

    pca = PCA()
    xRTrain = pca.fit_transform(scale(xTrain))
    xRTest = pca.fit_transform(scale(xTest))

    lmP = LinearRegression().fit(xRTrain, yTrain)
    R2CV = cross_val_score(lmP, xRTest, yTest, cv=10, scoring="r2").mean()
    error = -cross_val_score(lmP, xRTest, yTest, cv=10, scoring="neg_mean_squared_error").mean()

    print(R2CV)
    print("----" * 30)
    print(np.sqrt(error))

    scaler = StandardScaler().fit(xTrain, yTrain)
    xRTrain = scaler.transform(xTrain)
    xRTest = scaler.transform(xTest)

    mlpr = MLPRegressor().fit(xTrain, yTrain)

    R2CV = cross_val_score(mlpr, xRTest, yTest, cv=10, scoring="r2").mean()
    error = -cross_val_score(mlpr, xRTest, yTest, cv=10, scoring="neg_mean_squared_error").mean()

    print(R2CV)
    print("----" * 30)
    print(np.sqrt(error))

    lj = LogisticRegression(solver="liblinear").fit(xTrain, yTrain)
    gnb = GaussianNB().fit(xTrain, yTrain)
    knnc = KNeighborsClassifier().fit(xTrain, yTrain)
    print(knnc)
    cartc = DecisionTreeClassifier(random_state=42).fit(xTrain, yTrain)
    rfc = RandomForestClassifier(random_state=42, verbose=False).fit(xTrain, yTrain)
    gbmc = GradientBoostingClassifier(verbose=False).fit(xTrain, yTrain)
    xgbc = XGBClassifier().fit(xTrain, yTrain)
    lgbmc = LGBMClassifier().fit(xTrain, yTrain)
    catbc = CatBoostClassifier(verbose=False).fit(xTrain, yTrain)

    modelsc = [lj, gnb, knnc, cartc, rfc, gbmc, xgbc, lgbmc, catbc]

    for model in modelsc:
        name = model.__class__.__name__
        predict = model.predict(xTest)
        R2CV = cross_val_score(model, xTest, yTest, cv=10, verbose=False).mean()
        error = -cross_val_score(model, xTest, yTest, cv=10, scoring="neg_mean_squared_error", verbose=False).mean()
        print(name + ": ")
        print("-" * 10)
        print(accuracy_score(yTest, predict))
        print(R2CV)
        print(np.sqrt(error))
        print("-" * 30)

    r = pd.DataFrame(columns=["MODELS", "R2CV"])
    for model in modelsc:
        name = model.__class__.__name__
        R2CV = cross_val_score(model, xTest, yTest, cv=10, verbose=False).mean()
        result = pd.DataFrame([[name, R2CV * 100]], columns=["MODELS", "R2CV"])
        r = r.append(result)

    figure = plt.figure(figsize=(20, 8))
    sns.barplot(x="R2CV", y="MODELS", data=r, color="k")
    plt.xlabel("R2CV")
    plt.ylabel("MODELS")
    plt.xlim(0, 100)
    plt.title("MODEL ACCURACY COMPARISON")
    plt.show()

    # https: // scikit - learn.org / stable / modules / generated / sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler().fit(xTrain, yTrain)
    xRTrain = scaler.transform(xTrain)
    xRTest = scaler.transform(xTest)

    mlpc = MLPClassifier().fit(xRTrain, yTrain)
    predict = mlpc.predict(xRTest)

    R2CV = cross_val_score(mlpc, xRTest, yTest, cv=10).mean()
    print(R2CV)
    error = mean_squared_error(yTest, predict)
    print(np.sqrt(error))

    # params = {"n_estimators": [100, 500, 1000, 2000],
    #           "subsample": [0.6, 0.8, 1.0],
    #           "max_depth": [3, 4, 5, 6],
    #           "learning_rate": [0.1, 0.01, 0.02, 0.05],
    #           "min_child_samples": [5, 10, 20]}

    params = {"n_estimators": [100],
              "subsample": [0.6, 0.8],
              "max_depth": [3, 4, 5],
              "learning_rate": [0.1, 0.01, 0.02],
              "min_child_samples": [5, 10]}

    cv = GridSearchCV(lgbmc, params, cv=10, verbose=False, n_jobs=-1).fit(xTrain, yTrain)
    print(cv.best_params_)
    print(cv.best_score_)

    lgbmc_tuned = LGBMClassifier(learning_rate=0.01, max_depth=5, min_child_samples=10,
                                n_estimators=100, subsample=0.6).fit(xTrain, yTrain)

    R2CV_tuned = cross_val_score(lgbmc_tuned, xTest, yTest, cv=10).mean()
    print(R2CV_tuned)
    error_tuned = -cross_val_score(lgbmc_tuned, xTest, yTest, cv=10, scoring="neg_mean_squared_error").mean()
    print(np.sqrt(error_tuned))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # process_data( '../../../data/AHS_Woman_21_Odisha.csv')
    process_data( '../../../data/AHS_Woman_18_Assam.csv')
