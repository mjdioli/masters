from urllib import response
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path
import json
import pickle
import collections

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# TODO figure out a good set of hyperparameters and a response on how to tune them
# Perhaps irrelevant since we are looking at difference in performance due to missing data?
MODELS = {"log_reg": LogisticRegression(random_state=0, max_iter=500),
               "lin_reg": LinearRegression(),
               "svm": LinearSVC(random_state=0, tol=1e-5),
               "knn": KNeighborsClassifier(n_neighbors=3),
               "rf_cat": RandomForestClassifier(max_depth=4, random_state=0),
               "rf_cont": RandomForestRegressor(max_depth=4, random_state=0)}

IMPUTATIONS = ["cca", "mean", "mice_reg", "mice_def", "reg"]

SAVEPATH = "experiments/"

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)


def save(stem, filepath, data):
    if not os.path.isdir(Path(stem)):
        os.mkdir(Path(stem))

    if filepath.split(".")[-1] == "png":
        pass
    elif filepath.split(".")[-1] == "json":
        with open(Path(stem+filepath), 'w') as f:
            json.dump(data, f)
    elif filepath.split(".")[-1] == "pickle":
        with open(Path(stem+filepath), 'wb') as f:
            pickle.dump(data, f)

def sigmoid(x, alpha):
    z = np.exp(-x+alpha)
    sig = 1 / (1 + z)
    return sig

def splitter(data, response= "score_text"):
    x = data.drop(response, axis = 1)
    y = data[response]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    x_train[response]=y_train
    x_test[response]=y_test
    return {"train":x_train, "test": x_test}

def compas_cleaning(df):
    new_df = df.dropna()
    new_df = new_df[(new_df["days_b_screening_arrest"]<=30)&
                    (new_df["days_b_screening_arrest"]>=-30)&
                    (new_df["is_recid"]!=-1)&
                    (new_df["c_charge_degree"]!="O")]
    new_df["length_of_stay"] = ((pd.to_datetime(df["c_jail_out"])-pd.to_datetime(df["c_jail_in"])).dt.days)
    new_df["length_of_stay"] = new_df["length_of_stay"].astype(int)
    
    #Perhaps limit dataset to only black and white participants
    new_df["is_Caucasian"] = new_df["race"].apply(lambda x: 1 if x=="Caucasian" else 0)
    new_df.drop(labels = ["c_jail_out", "c_jail_in", "days_b_screening_arrest", "is_recid", "race"],axis = 1, inplace = True)
    if "v_score_text" in new_df.columns:
        new_df.columns = ["score_text" if col == "v_score_text" else col for col in new_df.columns]
    new_df["score_text"] = new_df["score_text"].apply(lambda x: 0 if x=="High" else 1)
    new_df = pd.get_dummies(new_df, 
                            columns = ["c_charge_degree",
                                        "age_cat",
                                        "sex"],
                            drop_first=True)
    return new_df


def load_compas():
    cols = ["days_b_screening_arrest",
    "is_recid",
    "c_charge_degree",
    "c_jail_out",
    "c_jail_in",
    "age_cat",
    "race",
    "sex",
    "two_year_recid",
    ]
    recid = pd.read_csv("./compas_recid.csv", usecols=cols+["score_text"])
    violent_recid = pd.read_csv("./compas_violent_recid.csv", usecols=cols+["v_score_text"])
    return {"standard": splitter(compas_cleaning(recid)), "violent": splitter(compas_cleaning(violent_recid))}

def load_synthetic(ver = "recid"):
    if ver == "recid":
        size = 6000
        length_of_stay = np.random.normal(14, 46, size = size)
        two_year_recid = np.random.binomial(1,0.455, size = size)
        is_Caucasian = np.random.binomial(1,0.34, size = size)
        charge_degree_M = np.random.binomial(1,0.3567, size = size)
        age_greater_than_45 = np.random.binomial(1,0.209, size = size)
        age_less_than_25  = np.random.binomial(1,0.218, size = size)
        sex_male = np.random.binomial(1,0.809, size = size)
        score_text = np.around(sigmoid((length_of_stay*(0.1)+two_year_recid+is_Caucasian+charge_degree_M+
        age_greater_than_45+age_less_than_25+sex_male), alpha = -0.5)).astype(int)
        synth_cat = pd.DataFrame({"score_text": score_text, "length_of_stay":length_of_stay,
            "two_year_recid":two_year_recid, "is_Caucasian":is_Caucasian, "c_charge_degree_M":charge_degree_M,
            "age_greater_than_45":age_greater_than_45, "age_less_than_25":age_less_than_25, "sex_Male":sex_male})
        synth_cat_test = synth_cat.iloc[:round(0.333*size),:]
        synth_cat_train = synth_cat.iloc[round(0.333*size):,:]
    elif ver == "simple":
        size = 50000
        x_1 = np.random.normal(40, 10, size = size)
        x_2 = np.random.binomial(1,0.65, size = size)
        y = np.around(sigmoid(x_1+x_2*20, alpha = 50)).astype(int)
        synth_cat = pd.DataFrame({"y": y, "x_1":x_1, "x_2":x_2})
        synth_cat_test = synth_cat.iloc[:round(0.333*size),:]
        synth_cat_train = synth_cat.iloc[round(0.333*size):,:]
    else:
        size = 4000
        length_of_stay = np.random.normal(12.16, 50.93, size = size)
        two_year_recid = np.random.binomial(1,0.16, size = size)
        is_Caucasian = np.random.binomial(1,0.3629, size = size)
        charge_degree_M = np.random.binomial(1,0.3997, size = size)
        age_greater_than_45 = np.random.binomial(1,0.2373, size = size)
        age_less_than_25  = np.random.binomial(1,0.1905, size = size)
        sex_male = np.random.binomial(1,0.79, size = size)
        score_text = np.around(sigmoid((length_of_stay*(0.1)+two_year_recid+is_Caucasian+charge_degree_M+
        age_greater_than_45+age_less_than_25+sex_male), alpha = -4.5)).astype(int)
        synth_cat = pd.DataFrame({"score_text": score_text, "length_of_stay":length_of_stay,
            "two_year_recid":two_year_recid, "is_Caucasian":is_Caucasian, "c_charge_degree_M":charge_degree_M,
            "age_greater_than_45":age_greater_than_45, "age_less_than_25":age_less_than_25, "sex_Male":sex_male})
        synth_cat_test = synth_cat.iloc[:round(0.333*size),:]
        synth_cat_train = synth_cat.iloc[round(0.333*size):,:]

    return {"test":synth_cat_test, "train": synth_cat_train}


def spd(pred, protected_class):
    """
    Equation: |P(Y_pred = y | Z = 1) - P(Y_pred = y | Z = 0)|
    Assumes protected_class is 0/1 binary"""
    z_1 = [y for y, z in zip(pred, np.array(protected_class)) if z == 1]
    z_0 = [y for y, z in zip(pred, np.array(protected_class)) if z == 0]
    if len(z_1)+len(z_0)!=len(pred):
        print("NOT EQUAL")
    return abs(sum(z_1)/len(z_1)-sum(z_0)/len(z_0))


def equalised_odds(pred, prot, true):
    """
    Equation: |P(Y_pred = y_pred | Y_true = y_true, Z = 1) - P(Y_pred = y_pred | Y_true = y_true, Z = 0)|
    Assumes prot is 0/1 binary"""
    z1_y0 = [y_hat for y_hat, z, y in zip(
        pred, prot, true) if z == 1 and y == 0]
    z0_y0 = [y_hat for y_hat, z, y in zip(
        pred, prot, true) if z == 0 and y == 0]
    z1_y1 = [y_hat for y_hat, z, y in zip(
        pred, prot, true) if z == 1 and y == 1]
    z0_y1 = [y_hat for y_hat, z, y in zip(
        pred, prot, true) if z == 0 and y == 1]
    return {"Y=1": abs(sum(z1_y1)/len(z1_y1)-sum(z0_y1)/len(z0_y1)),
            "Y=0": abs(sum(z1_y0)/len(z1_y0)-sum(z0_y0)/len(z0_y0))}


def predictive_parity(pred, prot, true):
    """
    Equation: |P(Y_true = 1 | Y_pred = y_pred, Z = 1) - P(Y_true = 1 | Y_pred = y_pred, Z = 0)|
    Assumes prot is 0/1 binary and that y=1 is what we care about"""
    z1_yhat1 = [y for y_hat, z, y in zip(
        pred, prot, true) if z == 1 and y_hat == 1]
    z0_yhat1 = [y for y_hat, z, y in zip(
        pred, prot, true) if z == 0 and y_hat == 1]
    try:
        z1 = sum(z1_yhat1)/len(z1_yhat1)
        z0 = sum(z0_yhat1)/len(z0_yhat1)
        return abs(z1-z0)
    except Exception as e:
        #print("Exception: ", e)
        return -1
    

def confusion_matrix(true, pred):
    # Assumes numpy arrays(
    try:
        tpr = sum([1 if t == p and p == 1 else 0 for t,
                  p in zip(true, pred)])/(sum(true))
    except:
        tpr = 0
        #print("true", sum(true))
        #print("pred", sum(pred))

    try:
        tnr = sum([1 if t == p and p == 0 else 0 for t,
                  p in zip(true, pred)])/(len(true)-sum(true))
    except:
        tnr = 0
        #print("true", sum(true))
        #print("pred", sum(pred))
    fpr = 1-tnr
    fnr = 1-tpr
    #Old return structure. Converted to vanilla dict for json compatibility
    #return pd.DataFrame({"Predicted true": [tpr, fpr],
    #                     "Predicted false": [fnr, tnr]}, index=["Is true", "Is false"])
    return {"Predicted true": [tpr, fpr],
                "Predicted false": [fnr, tnr]}



def data_remover_cat(full_data, missing_col, missing_pct, missing="mar"):
    # Missing_pct is in the range 0 to 100
    data = full_data.copy()

    if missing == "mar":
        x = data.drop(missing_col, axis=1)
        if data[missing_col].nunique() == 2:
            clf = LogisticRegression(random_state=0).fit(x, data[missing_col])
            preds = clf.predict_proba(x)[:, 1]
        else:
            clf = LinearRegression().fit(x, data[missing_col])
            preds = clf.predict(x)
        # print(preds)
        lower_percentile = np.percentile(preds, missing_pct//2)
        upper_percentile = np.percentile(preds, 100-missing_pct//2)
        """print("lower", lower_percentile,
            "upper", upper_percentile,
            "filtered", preds[(preds>=lower_percentile)&(preds<=upper_percentile)])
        
        #print("Mask", sum((data["preds"]<= lower_percentile)| (data["preds"]>= upper_percentile)))"""
        data["preds"] = preds
        data[missing_col] = data[missing_col].mask((data["preds"] <= lower_percentile) | (data["preds"] >= upper_percentile),
                                                   other=np.nan)
        data.drop("preds", axis=1, inplace=True)

    else:
        mcar = np.random.binomial(n=1, p=missing_pct/100, size=len(data))
        data["missing"] = [np.nan if m == 1 else 0 for m in mcar]
        data[missing_col] = data[missing_col].mask(data["missing"] == np.nan,
                                                   other=np.nan)
        data.drop("missing", axis=1, inplace=True)
    return data


#TODO unfinished
def regression_imputer(full_data, missing):
    # Check if data is binary categorical or continuous.
    cca = full_data.dropna()
    if cca[missing].nunique() == 2:
        pass
    else:
        pass

@ignore_warnings(category=ConvergenceWarning)
def impute(dataframe, missing_col, impute="cca"):
    data = dataframe.copy()
    if impute == "cca":
        data.dropna(axis=0, inplace=True)
    elif impute == "mean":
        if data[missing_col].nunique() == 2:
            mode = data[missing_col].mode(dropna=True)[0]
            data[missing_col] = data[missing_col].fillna(mode)
        else:
            mean = data[missing_col].mean(skipna=True)
            data[missing_col] = data[missing_col].fillna(mean)
    elif impute == "reg":
        pass
    elif impute == "mice_def":
        imputer = IterativeImputer(random_state=0)
        imputer.fit(data)
        data = pd.DataFrame(imputer.transform(data), columns=data.columns)
        if data[missing_col].nunique() == 2:
            data[missing_col] = data[missing_col].round()
    elif impute == "mice_reg":
        if data[missing_col].nunique() == 2:
            model = LogisticRegression(random_state=0, max_iter=300)
            imputer = IterativeImputer(estimator=model, random_state=0)
            imputer.fit(data)
            data = pd.DataFrame(imputer.transform(data), columns=data.columns)
        else:
            model = LinearRegression()
            imputer = IterativeImputer(estimator=model, random_state=0)
            imputer.fit(data)
            data = pd.DataFrame(imputer.transform(data), columns=data.columns)
    return data

#TODO Note down overall accuracy scores in a json.
@ignore_warnings(category=ConvergenceWarning)
def test_bench(train, test, pred: str, missing: str, sensitive: str, pred_var_type: str = "cat",
    percentiles = None):

    # sensitive var
    class_0_test = test[test[sensitive] == 0]
    class_1_test = test[test[sensitive] == 1]

    # print("class_0",sum(class_0_test[pred]))
    # print("class_1",sum(class_1_test[pred]))
    results = {}
    if percentiles is None:
        percentiles = [i for i in range(1, 16)]+[j for j in range(20, 100, 10)]

    # TODO add xgboost, neural network
    if pred_var_type == "cat":
        models = ["log_reg", "rf_cat", "svm", "knn"]
    else:
        models = ["lin_reg", "rf_cont"]
    # Run with full data

    results = {"mar": {"spd": {m: {i: [] for i in IMPUTATIONS} for m in models}, "eo": {m: {i: [] for i in IMPUTATIONS} for m in models}, "pp": {m: {i: [] for i in IMPUTATIONS} for m in models}},
                        "mcar": {"spd": {m: {i: [] for i in IMPUTATIONS} for m in models}, "eo": {m: {i: [] for i in IMPUTATIONS} for m in models}, "pp": {m: {i: [] for i in IMPUTATIONS} for m in models}},
                        "percentiles": percentiles}

    for m in tqdm(models):
        predictions_0 = MODELS[m].fit(train.drop(pred, axis=1),
                                           train[pred]).predict(class_0_test.drop(pred, axis=1))
        predictions_1 = MODELS[m].fit(train.drop(pred, axis=1),
                                           train[pred]).predict(class_1_test.drop(pred, axis=1))
        results[m+"_0"] = confusion_matrix(class_0_test[pred], predictions_0)
        results[m+"_1"] = confusion_matrix(class_1_test[pred], predictions_1)
        for p in percentiles:
            for imp in ["cca", "mice_def", "mean"]:
                # TPR, FPR, TNR, FNR data
                data_mcar = impute(data_remover_cat(
                    train, missing, p, missing="mcar"), missing, impute=imp)
                predictions_0 = MODELS[m].fit(data_mcar.drop(pred, axis=1),
                                                   data_mcar[pred]).predict(class_0_test.drop(pred, axis=1))
                predictions_1 = MODELS[m].fit(data_mcar.drop(pred, axis=1),
                                                   data_mcar[pred]).predict(class_1_test.drop(pred, axis=1))
                results[m+"_mcar_"+imp+"_" +
                        str(p)+"_0"] = confusion_matrix(class_0_test[pred], predictions_0)
                results[m+"_mcar_"+imp+"_" +
                        str(p)+"_1"] = confusion_matrix(class_1_test[pred], predictions_1)

                # Fairness metrics
                y_hat = MODELS[m].fit(data_mcar.drop(pred, axis=1),
                                           data_mcar[pred]).predict(test.drop(pred, axis=1))
                results["mcar"]["spd"][m][imp].append(
                    spd(y_hat, test[sensitive]))
                results["mcar"]["eo"][m][imp].append(
                    equalised_odds(y_hat, test[sensitive], test[pred]))
                results["mcar"]["pp"][m][imp].append(
                    predictive_parity(y_hat, test[sensitive], test[pred]))

                # TPR, FPR, TNR, FNR data
                data_mar = impute(data_remover_cat(
                    train, missing, p, missing="mar"), missing, impute=imp)
                
                try:
                    predictions_0 = MODELS[m].fit(data_mar.drop(pred, axis=1),
                                                    data_mar[pred]).predict(class_0_test.drop(pred, axis=1))
                except Exception as e:
                    print("Exception: ", e)
                    print("params: ", str(p)+imp)
                    print("head: ", data_mar.head())
                    print("sum: ", data_mar.sum()-len(data_mar))
                predictions_1 = MODELS[m].fit(data_mar.drop(pred, axis=1),
                                                   data_mar[pred]).predict(class_1_test.drop(pred, axis=1))
                results[m+"_mar_"+imp+"_" +
                        str(p)+"_0"] = confusion_matrix(class_0_test[pred], predictions_0)
                results[m+"_mar_"+imp+"_" +
                        str(p)+"_1"] = confusion_matrix(class_1_test[pred], predictions_1)

                # Fariness metrics
                y_hat = MODELS[m].fit(data_mar.drop(pred, axis=1),
                                           data_mar[pred]).predict(test.drop(pred, axis=1))
                results["mar"]["spd"][m][imp].append(
                    spd(y_hat, test[sensitive]))
                results["mar"]["eo"][m][imp].append(
                    equalised_odds(y_hat, test[sensitive], test[pred]))
                results["mar"]["pp"][m][imp].append(
                    predictive_parity(y_hat, test[sensitive], test[pred]))
    try:
        save("./data/", "testresults.pickle", results)
    except Exception as e:
        print("Couldn't save data with exception: ", e)
    return results


#TODO Add filename thingy
def plotting_cf(models, correctives, results, key = None):
    if key is None:
        if not os.path.isdir(Path(SAVEPATH)):
            os.mkdir(Path(SAVEPATH))
            savepath = SAVEPATH
    else:
        if not os.path.isdir(Path(SAVEPATH+key)):
            os.mkdir(Path(SAVEPATH+key))
            savepath = SAVEPATH+key
    for m in models:
        for c in correctives:
            tpr_mar = {"0": {}, "1": {}}
            tnr_mar = {"0": {}, "1": {}}
            tpr_mcar = {"0": {}, "1": {}}
            tnr_mcar = {"0": {}, "1": {}}
            for key, value in results.items():
                if m+"_mar" in key and c in key:
                    if key[-1] == "0":
                        tpr_mar["0"][int(key.split("_")[-2])
                                     ] = value.iloc[0, 0]
                        tnr_mar["0"][int(key.split("_")[-2])
                                     ] = value.iloc[1, 1]
                    else:
                        tpr_mar["1"][int(key.split("_")[-2])
                                     ] = value.iloc[0, 0]
                        tnr_mar["1"][int(key.split("_")[-2])
                                     ] = value.iloc[1, 1]
                elif m+"_mcar" in key and c in key:
                    try:
                        if key[-1] == "0":
                            tpr_mcar["0"][int(key.split("_")[-2])
                                          ] = value.iloc[0, 0]
                            tnr_mcar["0"][int(key.split("_")[-2])
                                          ] = value.iloc[1, 1]
                        else:
                            tpr_mcar["1"][int(key.split("_")[-2])
                                          ] = value.iloc[0, 0]
                            tnr_mcar["1"][int(key.split("_")[-2])
                                          ] = value.iloc[1, 1]
                    except Exception as e:
                        print("key", key, "exception", e)
            tpr_mar = collections.OrderedDict(sorted(tpr_mar.items()))

            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            #TODO add thing that makes plots bigger
            plt.plot(list(tpr_mar["0"].keys()), list(
                tpr_mar["0"].values()), label=m+"TPR MAR class 0")
            plt.plot(list(tnr_mar["0"].keys()), list(
                tnr_mar["0"].values()), label=m+"TNR MAR class 0")
            plt.plot(list(tpr_mar["1"].keys()), list(
                tpr_mar["1"].values()), label=m+"TPR MAR class 1")
            plt.plot(list(tnr_mar["1"].keys()), list(
                tnr_mar["1"].values()), label=m+"TNR MAR class 1")
            plt.title(m+"_"+c+"_MAR")
            plt.xlabel("Missingness percent")
            plt.ylabel("Accuracy")
            plt.ylim(0.0, 1.01)
            plt.legend()
            plt.savefig(Path(savepath+m+"_"+c+"_MAR.png"))
            plt.clf()

            plt.plot(list(tpr_mcar["0"].keys()), list(
                tpr_mcar["0"].values()), label=m+"TPR MCAR class 0")
            plt.plot(list(tnr_mcar["0"].keys()), list(
                tnr_mcar["0"].values()), label=m+"TNR MCAR class 0")
            plt.plot(list(tpr_mcar["1"].keys()), list(
                tpr_mcar["1"].values()), label=m+"TPR MCAR class 1")
            plt.plot(list(tnr_mcar["1"].keys()), list(
                tnr_mcar["1"].values()), label=m+"TNR MCAR class 1")
            plt.title(m+"_"+c+"_MCAR")
            plt.xlabel("Missingness percent")
            plt.ylabel("Accuracy")
            plt.ylim(0.0, 1.01)
            plt.legend()
            plt.savefig(Path(SAVEPATH+m+"_"+c+"_MCAR.png"))
            plt.clf()


def plotting_others(results, key = None):
    if key is None:
        if not os.path.isdir(Path(SAVEPATH)):
            os.mkdir(Path(SAVEPATH))
        savepath = SAVEPATH
    else:
        if not os.path.isdir(Path(SAVEPATH+key)):
            os.mkdir(Path(SAVEPATH+key))
        savepath = SAVEPATH+key
    for missingness, data in results.items():
        if missingness != "mar" and missingness != "mcar":
            continue
        for metric, res in data.items():
            for model, imps in res.items():
                for imputation, vals in imps.items():
                    if len(vals) == 0:
                        continue
                    if isinstance(vals[0], dict):
                        y_0 = np.zeros(len(vals))
                        y_1 = np.zeros(len(vals))

                        for i,dictio in enumerate(vals):
                            y_0[i] = dictio["Y=0"]
                            y_1[i] = dictio["Y=1"]

                        fig = plt.gcf()
                        fig.set_size_inches(18.5, 10.5)
                        plt.plot(list(results["percentiles"]), list(
                        y_0), label=model+"_"+imputation+"Y=0")
                        plt.plot(list(results["percentiles"]), list(
                        y_1), label=model+"_"+imputation+"Y=1")
                        plt.title(model+"_"+imputation+"_"+metric+"_"+missingness)
                        plt.xlabel("Missingness percent")
                        plt.ylabel(metric)
                        plt.legend()
                        

                    else:
                        fig = plt.gcf()
                        fig.set_size_inches(18.5, 10.5)
                        plt.plot(list(results["percentiles"]), list(
                            vals), label=model+"_"+imputation)
                        plt.title(model+"_"+imputation+"_"+metric+"_"+missingness)
                        plt.xlabel("Missingness percent")
                        plt.ylabel(metric)
                        plt.legend()
                plt.savefig(Path(savepath+model+"_"+imputation+"_"+metric+"_"+missingness+".png"))
                plt.clf()
