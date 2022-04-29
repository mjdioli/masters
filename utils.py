from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
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


def save(stem, filepath, data):
    if not os.path.isdir(Path(stem)):
        os.mkdir(Path(stem))

    if filepath.split(".")[-1] == "png":
        pass
    elif filepath.split(".")[-1] == "json":
        with open(Path(stem+filepath), 'wb') as f:
            json.dump(data, f)
    elif filepath.split(".")[-1] == "pickle":
        with open(Path(stem+filepath), 'wb') as f:
            pickle.dump(data, f)


def sigmoid(x, alpha):
    z = np.exp(-x+alpha)
    sig = 1 / (1 + z)
    return sig

def spd(pred, protected_class):
    """Assumes z is 0/1 binary"""
    z_1 = [y for y, z in zip(pred, np.array(protected_class)) if z == 1]
    z_0 = [y for y, z in zip(pred, np.array(protected_class)) if z == 0]
    return abs(sum(z_1)/len(z_1)-sum(z_0)/len(z_0))


def equalised_odds(pred, prot, true):
    """Assumes prot is 0/1 binary"""
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
    """Assumes prot is 0/1 binary and that y=1 is what we care about"""
    z1_yhat1 = [y for y_hat, z, y in zip(
        pred, prot, true) if z == 1 and y == 1 and y_hat == 1]
    z0_yhat1 = [y for y_hat, z, y in zip(
        pred, prot, true) if z == 0 and y == 1 and y_hat == 1]
    return abs(sum(z1_yhat1)/len(z1_yhat1)-sum(z0_yhat1)/len(z0_yhat1))

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
    return pd.DataFrame({"Predicted true": [tpr, fpr],
                         "Predicted false": [fnr, tnr]}, index=["Is true", "Is false"])


def spd(pred, protected_class):
    """Assumes z is 0/1 binary"""
    z_1 = [y for y, z in zip(pred, np.array(protected_class)) if z == 1]
    z_0 = [y for y, z in zip(pred, np.array(protected_class)) if z == 0]
    return abs(sum(z_1)/len(z_1)-sum(z_0)/len(z_0))


def equalised_odds(pred, prot, true):
    """Assumes prot is 0/1 binary"""
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
    """Assumes prot is 0/1 binary and that y=1 is what we care about"""
    z1_yhat1 = [y for y_hat, z, y in zip(
        pred, prot, true) if z == 1 and y == 1 and y_hat == 1]
    z0_yhat1 = [y for y_hat, z, y in zip(
        pred, prot, true) if z == 0 and y == 1 and y_hat == 1]
    return abs(sum(z1_yhat1)/len(z1_yhat1)-sum(z0_yhat1)/len(z0_yhat1))


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


def impute(dataframe, missing_col, impute="cca"):
    data = dataframe.copy()
    if impute == "cca":
        data.dropna(axis=0, inplace=True)
    elif impute == "mean":
        if data[missing_col].nunique() == 2:
            #print("nans", data[missing_col].isna().sum())
            mode = data[missing_col].mode(dropna=True)[0]
            #print("mode", mode, "END")
            data[missing_col] = data[missing_col].fillna(mode)
            #print("nans", data[missing_col].isna().sum())
        else:
            mean = data[missing_col].mean(skipna=True)
            data[missing_col] = data[missing_col].fillna(mean)
    elif impute == "reg":
        pass
    elif impute == "mice_def":
        imputer = IterativeImputer(random_state=0)
        imputer.fit(data)
        data = pd.DataFrame(imputer.transform(data), columns=data.columns)
        # print(data[missing_col].unique())
        if data[missing_col].nunique() == 2:
            data[missing_col] = data[missing_col].round()
        # print(data[missing_col].unique())
    elif impute == "mice_reg":
        if data[missing_col].nunique() == 2:
            model = LogisticRegression(random_state=0, max_iter=300)
            imputer = IterativeImputer(estimator=model, random_state=0)
            imputer.fit(data)
            data = pd.DataFrame(imputer.transform(data), columns=data.columns)
            # print(data[missing_col].unique())
        else:
            model = LinearRegression()
            imputer = IterativeImputer(estimator=model, random_state=0)
            imputer.fit(data)
            data = pd.DataFrame(imputer.transform(data), columns=data.columns)
    return data


def test_bench(train, test, pred: str, missing: str, sensitive: str, pred_var_type: str = "cat"):

    # sensitive var
    class_0_test = test[test[sensitive] == 0]
    class_1_test = test[test[sensitive] == 1]

    # print("class_0",sum(class_0_test[pred]))
    # print("class_1",sum(class_1_test[pred]))
    results = {}
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
                predictions_0 = MODELS[m].fit(data_mar.drop(pred, axis=1),
                                                   data_mar[pred]).predict(class_0_test.drop(pred, axis=1))
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


def plotting_cf(models, correctives, results):
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
            plt.show()

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
            plt.show()


def plotting_others(results):
    for missingness, data in results.items():
        if missingness != "mar" and missingness != "mcar":
            continue
        for metric, res in data.items():
            for model, imps in res.items():
                for imputation, vals in imps.items():
                    if len(vals) == 0:
                        continue
                    plt.plot(list(results["percentiles"]), list(
                        vals), label=model+"_"+imputation)
                    plt.title(model+"_"+imputation+"_"+metric+"_"+missingness)
                    plt.xlabel("Missingness percent")
                    plt.ylabel(metric)
                    plt.legend()
                plt.show()
