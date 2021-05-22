import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
import shap as shap
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from scipy.stats import levene
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.ensemble import  GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import  CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import  accuracy_score
import dice_ml
from IPython.display import display
from com.shariful.prediction.scrubbing import scrub

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
    sm.qqplot(data['mother_age_when_baby_was_born'], line='s')
    plt.show()

def makeData():
   odisha_path='../../../data/used_data/AHS_Woman_21_Odisha.csv'
   rajsthan_path='../../../data/used_data/AHS_Woman_08_Rajasthan.csv'
   assam_path='../../../data/used_data/AHS_Woman_18_Assam.csv'
   jharkhand_path='../../../data/used_data/AHS_Woman_20_Jharkhand.csv'
   chattisgarh_path='../../../data/used_data/AHS_Woman_22_Chhattisgarh.csv'

   odisha_data = pd.read_csv(odisha_path, delimiter='|', nrows=1000)
   rajsthan_data = pd.read_csv(rajsthan_path, delimiter='|', nrows=1000)
   assam_data = pd.read_csv(assam_path, delimiter='|', nrows=1000)
   jharkhand_data = pd.read_csv(jharkhand_path, delimiter='|', nrows=1000)
   chattisgarh_data = pd.read_csv(chattisgarh_path, delimiter='|', nrows=1000)
   frames=[odisha_data,rajsthan_data,assam_data,jharkhand_data,chattisgarh_data]
   data=pd.concat(frames)
   data.to_csv('../../../data/used.csv', index=False)

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
    oversample = SMOTE()
    xTrain, yTrain = oversample.fit_resample(xTrain, yTrain)
    xTest, yTest = oversample.fit_resample(xTest, yTest)

    gnb = GaussianNB().fit(xTrain, yTrain)
    knnc = KNeighborsClassifier().fit(xTrain, yTrain)
    rfc = RandomForestClassifier(random_state=42, verbose=False).fit(xTrain, yTrain)
    gbmc = GradientBoostingClassifier(verbose=False).fit(xTrain, yTrain)
    xgbc = XGBClassifier().fit(xTrain, yTrain)
    lgbmc = LGBMClassifier( num_leaves=30).fit(xTrain, yTrain)
    catbc = CatBoostClassifier(verbose=False).fit(xTrain, yTrain)

    modelsc = [
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
    sns.set(font_scale=2)
    sns.barplot(x="Cross Validation Score", y="Trained Models", data=r, color="k",orient="h")
    plt.xlabel("Cross Validation Score")
    plt.ylabel("Trained Models")
    plt.xlim(0, 100)
    plt.title("MODEL ACCURACY COMPARISON")
    plt.subplots_adjust(left=0.3)
    plt.show()

    explainer = shap.TreeExplainer(rfc)
    shap_values = explainer.shap_values(xTest)
    shap.initjs()
    figure = plt.figure(figsize=(20, 12))

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
    scaler = StandardScaler().fit(xTrain, yTrain)
    xRTrain = scaler.transform(xTrain)
    xRTest = scaler.transform(xTest)
    mlpc = MLPClassifier().fit(xRTrain, yTrain)
    predict = mlpc.predict(xTest)
    R2CV = cross_val_score(mlpc, xRTest, yTest, cv=10).mean()
    print(R2CV)
    error = mean_squared_error(yTest, predict)
    print(np.sqrt(error))
    joblib.dump(rfc, '../../../data/saved_model/model.bpk')
    loaded_model=joblib.load('../../../data/saved_model/model.bpk',)
    error_tuned_new = -cross_val_score(loaded_model, xTest, yTest, cv=10, scoring="neg_mean_squared_error").mean()
    print(np.sqrt(error_tuned_new))


if __name__ == '__main__':
    process_data()
