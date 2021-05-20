
import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
import shap as shap
from imblearn.over_sampling import SMOTE
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
import missingno as msno
from pandas.core.ops import radd
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

import dice_ml

from IPython.display import display
from dice_ml.utils import helpers




import os
import pickle
from sklearn import preprocessing
from com.shariful.prediction.scrubbing import scrub

from collections import Counter


def variance(data):
    print("%.4f - %.4f " % levene(data["age"],
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
    # plt.hist(data)

    # sm.qqplot(data['age'], line='s')
    #     # plt.show()
    #     # plt.savefig('age_qq.png')
    # sm.qqplot(data['wt'], line='s')
    # plt.show()
    # plt.savefig('weight_qq.png')
    sm.qqplot(data['mother_age_when_baby_was_born'], line='s')
    plt.show()
    # plt.savefig('baby_age.png')




def makeData():
   odisha_path='../../../data/AHS_Woman_21_Odisha.csv'
   rajsthan_path='../../../data/AHS_Woman_08_Rajasthan.csv'
   assam_path='../../../data/AHS_Woman_18_Assam.csv'
   jharkhand_path='../../../data/AHS_Woman_20_Jharkhand.csv'
   chattisgarh_path='../../../data/AHS_Woman_22_Chhattisgarh.csv'

   odisha_data = pd.read_csv(odisha_path, delimiter='|', nrows=500)
   rajsthan_data = pd.read_csv(rajsthan_path, delimiter='|', nrows=500)
   assam_data = pd.read_csv(assam_path, delimiter='|', nrows=500)
   jharkhand_data = pd.read_csv(jharkhand_path, delimiter='|', nrows=500)
   chattisgarh_data = pd.read_csv(chattisgarh_path, delimiter='|', nrows=500)

   frames=[odisha_data,rajsthan_data,assam_data,jharkhand_data,chattisgarh_data]
   # frames=[odisha_data]
   data=pd.concat(frames)
   return data





def process_data():
    filterwarnings("ignore")
    data = makeData()
    data=scrub.scrubData(data)
    print(data.shape)
    print(data.columns)

    normality(data)

    x = data.drop("outcome_pregnancy", axis=1)
    y = data["outcome_pregnancy"]
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.20, random_state=1)

    # counter = Counter(yTrain)
    # print(counter)

    # y0=yTrain[yTrain.outcome_pregnancy==0]
    # y1=yTrain[yTrain.outcome_pregnancy==1]
    # plt.scatter(y0,y1)
    # plt.show
    # yTrain.plot(style='.')

    # pd.Series.plot(data=yTrain,kind='scatter')
    # plt.show

    # variance(data)


    # for label, n in counter.items():
    #     row_ix = np.where(yTrain == label)[0]
    #     plt.scatter(xTrain[row_ix, 0], xTrain.rural[row_ix, 1], label="test")
    # plt.legend()
    # plt.show()
    # print(counter)
    #balancing
    oversample = SMOTE()
    xTrain, yTrain = oversample.fit_resample(xTrain, yTrain)
    xTest, yTest = oversample.fit_resample(xTest, yTest)
    # counter = Counter(yTrain)
    # print(counter)


    # print(xTrain.describe())
    # print(yTrain.describe())


    #regression models
    # lm = LinearRegression().fit(xTrain, yTrain)
    # Partial_least_squares_regression = PLSRegression().fit(xTrain, yTrain)
    # ridge = Ridge().fit(xTrain, yTrain)
    # lasso = Lasso().fit(xTrain, yTrain)
    # elastic_net = ElasticNet().fit(xTrain, yTrain)
    # decision_tree = DecisionTreeRegressor(random_state=42).fit(xTrain, yTrain)
    # knn = KNeighborsRegressor().fit(xTrain, yTrain)
    # bagging = BaggingRegressor(random_state=42, bootstrap_features=True, verbose=False).fit(xTrain, yTrain)
    # random_forest_regressor = RandomForestRegressor(random_state=42, verbose=False).fit(xTrain, yTrain)
    # gradient_boosting = GradientBoostingRegressor(verbose=False).fit(xTrain, yTrain)
    # xg_boost = XGBRegressor().fit(xTrain, yTrain)
    # light_bm_regressor = LGBMRegressor().fit(xTrain, yTrain)
    # cat_boost_regressor = CatBoostRegressor(verbose=False).fit(xTrain, yTrain)
    # mlpr = MLPRegressor().fit(xTrain, yTrain)

    # perm = PermutationImportance(random_forest_regressor, random_state=1).fit(xTest, yTest)
    # eli5.show_weights(perm,feature_names = xTest.columns.tolist())

    # feature_names = [i for i in xTrain.columns if data[i].dtype in [np.float]]
    # tree_graph = tree.export_graphviz(decision_tree, out_file=None, feature_names=feature_names)
    # graphviz.Source(tree_graph)



    # models = [
    #     # lm,
    #     # Partial_least_squares_regression,
    #     # ridge,
    #     # lasso,
    #     # elastic_net,
    #     # decision_tree,
    #     knn,
    #     bagging,
    #     random_forest_regressor,
    #     gradient_boosting,
    #     xg_boost,
    #     light_bm_regressor,
    #     cat_boost_regressor,
    #     # mlpr
    # ]

    # for model in models:
    #     name = model.__class__.__name__
    #     R2CV = cross_val_score(model, xTest, yTest, cv=10, scoring="r2").mean()
    #     error = -cross_val_score(model, xTest, yTest, cv=10, scoring="neg_mean_squared_error").mean()
    #     print(name + ": ")
    #     print("-" * 10)
    #     print("r2 score : "+str(R2CV))
    #     print("neg_mean_squared_error : "+str(np.sqrt(error)))
    #     print("-" * 30)
    # r = pd.DataFrame(columns=["MODELS", "R2CV"])

    # for model in models:
    #     name = model.__class__.__name__
    #     R2CV = cross_val_score(model, xTest, yTest, cv=10, scoring="r2").mean()
    #     result = pd.DataFrame([[name, R2CV * 100]], columns=["MODELS", "R2CV"])
    #     r = r.append(result)


    # figure = plt.figure(figsize=(20, 8))
    # sns.barplot(x="R2CV", y="MODELS", data=r, color="k")
    # plt.xlabel("R2CV")
    # plt.ylabel("MODELS")
    # plt.xlim(-50, 100)
    # plt.title("MODEL ACCURACY COMPARISON")
    # plt.show()

    # ols = sm.OLS(yTrain, xTrain).fit()
    # print(ols.summary())

    #Principal component analysis
    # xPCA = StandardScaler().fit_transform(xTrain)
    # pca = PCA(n_components=2)
    # principalComponents = pca.fit_transform(scale(xTrain))
    #
    # principalDf = pd.DataFrame(data=principalComponents
    #                            , columns=['principal component 1', 'principal component 2'])
    #
    # finalDf = pd.concat([principalDf, yTrain[['outcome_pregnancy']]], axis=1)
    #
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel('Principal Component 1', fontsize=15)
    # ax.set_ylabel('Principal Component 2', fontsize=15)
    # ax.set_title('2 component PCA', fontsize=20)
    # targets = [1, 0]
    # colors = ['r', 'g']
    #
    #
    # for target, color in zip(targets, colors):
    #     indicesToKeep = finalDf['target'] == target
    #     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
    #                , finalDf.loc[indicesToKeep, 'principal component 2']
    #                , c=color
    #                , s=50)
    # ax.legend(targets)
    #
    # xRTest = pca.fit_transform(scale(xTest))




    gnb = GaussianNB().fit(xTrain, yTrain)
    knnc = KNeighborsClassifier().fit(xTrain, yTrain)
    rfc = RandomForestClassifier(random_state=42, verbose=False).fit(xTrain, yTrain)
    gbmc = GradientBoostingClassifier(verbose=False).fit(xTrain, yTrain)
    xgbc = XGBClassifier().fit(xTrain, yTrain)
    lgbmc = LGBMClassifier( num_leaves=30).fit(xTrain, yTrain)
    catbc = CatBoostClassifier(verbose=False).fit(xTrain, yTrain)


    # row_to_show = 5
    # data_for_prediction = xTest.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
    # data_for_prediction_array = data_for_prediction.values.reshape(1, -1)
    # rfc.predict_proba(data_for_prediction_ar   ray)


    # shap.plots.waterfall(shap_values[1],max_display=10)


    # figure = plt.figure(figsize=(20, 8))
    # sns.barplot(x="R2CV", y="MODELS", data=shap_values, color="k")
    # plt.xlabel("R2CV")
    # plt.ylabel("MODELS")
    # plt.xlim(-50, 100)
    # plt.title("SHap")
    # plt.show()



    modelsc = [
        # lj,
        # cartc,
        gnb,
        knnc,
        rfc,
        gbmc,
        xgbc,
        lgbmc,
        catbc]



    for model in modelsc:
        name = model.__class__.__name__
        predict = model.predict(xTest)
        R2CV = cross_val_score(model, xTest, yTest, cv=10, verbose=False).mean()
        error = -cross_val_score(model, xTest, yTest, cv=10, scoring="neg_mean_squared_error", verbose=False).mean()
        print(name + ": ")
        print("-" * 10)
        print("accuracy score : "+str(accuracy_score(yTest, predict)))
        print("r2 score : "+str(R2CV))
        print("neg_mean_squared_error : " +str(np.sqrt(error)))
        print("-" * 30)

    r = pd.DataFrame(columns=["Trained Models", "Cross Validation Score"])
    for model in modelsc:
        name = model.__class__.__name__
        R2CV = cross_val_score(model, xTest, yTest, cv=10, verbose=False).mean()
        result = pd.DataFrame([[name, R2CV * 100]], columns=["Trained Models", "Cross Validation Score"])
        r = r.append(result)

    figure = plt.figure(figsize=(20, 14))
    # plt.legend(fontsize="x-large")
    sns.set(font_scale=2)
    sns.barplot(x="Cross Validation Score", y="Trained Models", data=r, color="k",orient="h")
    plt.xlabel("Cross Validation Score")
    plt.ylabel("Trained Models")
    plt.xlim(0, 100)
    plt.title("MODEL ACCURACY COMPARISON")
    plt.subplots_adjust(left=0.3)
    plt.show()

    #SHAP

    explainer = shap.TreeExplainer(rfc)
    shap_values = explainer.shap_values(xTest)
    shap.initjs()
    figure = plt.figure(figsize=(20, 12))
    # shap.force_plot(explainer.expected_value, shap_values[0,], xTest.iloc[0,:])

    shap.summary_plot(shap_values, xTest,plot_type='bar',max_display=20)
    plt.subplots_adjust(left=0.3)
    plt.show()

    f = plt.figure(figsize=(20, 12))
    plt.subplots_adjust(left=0.3)
    shap.summary_plot(shap_values, xTest)
    f.savefig("summary_plot1.png", bbox_inches='tight', dpi=600)
    plt.show()


    #dice part
    d = dice_ml.Data(dataframe=data, continuous_features=['age','mother_age_when_baby_was_born','wt'], outcome_name='outcome_pregnancy')
    m = dice_ml.Model(model=rfc, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")
    e1 = exp.generate_counterfactuals(xTrain[0:1], total_CFs=2, desired_class="opposite")

    data=xTrain[0:1]
    data2=e1.cf_examples_list[0].final_cfs_df_sparse
    data=data.append(data2)

    data.to_csv('../../../data/counter.csv',index=False)




    display(e1.visualize_as_dataframe(show_only_changes=True))


    query_instance = xTrain[0:1]
    imp = exp.local_feature_importance(query_instance, total_CFs=10)
    print(imp.local_importance)

    query_instances = xTrain[0:100]
    imp = exp.global_feature_importance(query_instances)
    print(imp.summary_importance)

    # https: // scikit - learn.org / stable / modules / generated / sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler().fit(xTrain, yTrain)
    xRTrain = scaler.transform(xTrain)
    xRTest = scaler.transform(xTest)

    mlpc = MLPClassifier().fit(xRTrain, yTrain)
    predict = mlpc.predict(xTest)

    R2CV = cross_val_score(mlpc, xRTest, yTest, cv=10).mean()
    print(R2CV)
    error = mean_squared_error(yTest, predict)
    print(np.sqrt(error))

    # params = {"n_estimators": [100, 500, 1000, 2000],
    #           "subsample": [0.6, 0.8, 1.0],
    #           "max_depth": [3, 4, 5, 6],
    #           "learning_rate": [0.1, 0.01, 0.02, 0.05],
    #           "min_child_samples": [5, 10, 20]}

    # params = {"n_estimators": [100],
    #           "subsample": [0.6, 0.8],
    #           "max_depth": [3, 4, 5],
    #           "learning_rate": [0.1, 0.01, 0.02],
    #           "min_child_samples": [5, 10]}
    #
    # cv = GridSearchCV(lgbmc, params, cv=10, verbose=False, n_jobs=-1).fit(xTrain, yTrain)
    # print(cv.best_params_)
    # print(cv.best_score_)
    #
    # lgbmc_tuned = LGBMClassifier(learning_rate=0.01, max_depth=5, min_child_samples=10,
    #                             n_estimators=100, subsample=0.6).fit(xTrain, yTrain)
    #
    # R2CV_tuned = cross_val_score(lgbmc_tuned, xTest, yTest, cv=10).mean()
    # print(R2CV_tuned)
    # error_tuned = -cross_val_score(lgbmc_tuned, xTest, yTest, cv=10, scoring="neg_mean_squared_error").mean()
    # print(np.sqrt(error_tuned))

    joblib.dump(rfc, '../../../data/saved_model/model.bpk')

    loaded_model=joblib.load('../../../data/saved_model/model.bpk',)

    error_tuned_new = -cross_val_score(loaded_model, xTest, yTest, cv=10, scoring="neg_mean_squared_error").mean()
    print(np.sqrt(error_tuned_new))




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # process_data( '../../../data/AHS_Woman_21_Odisha.csv')
    process_data()
