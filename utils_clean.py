from fair_logistic_reg import FairLogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
import seaborn as sns
import utils


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
import copy
from itertools import permutations

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# TODO
MODELS = {"log_reg": LogisticRegression(random_state=0, max_iter=500),
          "svm": LinearSVC(random_state=0, tol=1e-5),
          "knn": KNeighborsClassifier(n_neighbors=3),
          "rf_cat": RandomForestClassifier(max_depth=4, random_state=0)}

#IMPUTATIONS = ["fair_reg_995", "cca", "fair_reg_985", "mean", "mice_def", "coldel"]# , "reg"]
IMPUTATIONS = ["fair_reg_95", "fair_reg_50", "log_reg", "cca", "mean", "mice_def", "coldel"]# , "reg"]
#IMPUTATIONS = ["cca", "mean"]#, "mice_def", "coldel"]# , "reg"]
METRICS = ["spd", "eosum", "acc", "tpr0", "tpr1", "tnr0", "tnr1", "tprd", "tnrd"]
IMPUTATION_COMBOS = [perm[0]+"|"+perm[1] for perm in permutations(IMPUTATIONS, 2)]

#missing_pct = [0, 25,50,75, 90, 95]
#RECID_ALPHA = [1000, 3.693, 2.84,1.94, 1.033,0.91]
#SIMPLE_ALPHA = [1000, 2.96, 1.94,0.979, 0.1,-0.41]

#missing_pct = [0, 10,25,50,75, 90]
RECID_ALPHA = [1000, 4.0, 3.693, 2.84,1.94, 1.033]
SIMPLE_ALPHA = [1000, 3.86, 2.96, 1.94,0.979, 0.1]
ADULT_ALPHA =  [1000, 7.267, 6.799, 6.23,5.5,4.92]


SAVEPATH = "experiments_final/"

NAME_KEYS = {"coldel": "Column deletion",
             "cca": "CCA",
             "mice_def": "Chained Eqaution",
             "mean": "Mean",
             "log_reg": "Logistic Regression",
             "svm": "SVM",
             "knn": "KNN",
             "rf_cat": "Random Forest",
             "fair_reg_99": "Fairness aware imputation lambda = 0.99",
             "fair_reg_95": "Fairness aware imputation lambda = 0.95",
             "fair_reg_75": "Fairness aware imputation lambda = 0.75",
             "fair_reg_50": "Fairness aware imputation lambda = 0.50"}

font = {'family': 'normal',
        'weight': 'bold',
        'size': 25}

plt.rc('font', **font)


def save(stem, filepath, data):
    if not os.path.isdir(Path(stem)):
        os.mkdir(Path(stem))

    if filepath.split(".")[-1] == "png":
        pass
    elif filepath.split(".")[-1] == "json":
        with open(Path(stem+filepath), 'w') as f:
            json.dump(data, f, indent=1)
    elif filepath.split(".")[-1] == "pickle":
        with open(Path(stem+filepath), 'wb') as f:
            pickle.dump(data, f)
            

#missingness_percentages = [0, 5,10,25,50,75]
def sigmoid(x, alpha):
    z = np.exp(-x+alpha)
    sig = 1 / (1 + z)
    return sig

def data_remover_cat(full_data, missing_col, alpha, noise, missing_pct = None, missingness="mar", robust = True):
    # Missing_pct is in the range 0 to 100
    
    data = full_data.copy()
    if alpha >100:
        return data
    if missingness == "mar":
        #print(data.drop(missing_col, axis = 1).sum(axis = 1), len(noise))
        ps = sigmoid(data.drop(missing_col, axis = 1).sum(axis = 1) + noise, alpha)
        
        if robust:
            data["miss"] = np.random.binomial(1,ps, size = len(ps))
        else:
            data["miss"] = np.around(ps)
        #print("PERCENT MISSING", data["miss"].sum())
        data[missing_col] = data[missing_col].mask((data["miss"]==1),
                                                    other=np.nan)
        data.drop("miss", axis=1, inplace=True)

    else:
        mcar = np.random.binomial(n=1, p=missing_pct/100, size=len(data))
        data["miss"] = [1 if m == 1 else 0 for m in mcar]
        data[missing_col] = data[missing_col].mask(data["miss"] == 1,
                                                   other=np.nan)
        data = data.drop("miss", axis=1)
    return data

@ignore_warnings(category=ConvergenceWarning)
def impute(dataframe,response, missing_col,sensitive_col, alpha, impute="cca"):
    #TODO add knn imputation
    data = dataframe.copy()
    if alpha >100 and impute != "coldel":
        return data
    impute_split = impute.split("_")
    #print(impute_split)
    if impute == "cca":
        data.dropna(axis=0, inplace=True)
        #print("Lost ", old_len-len(data), " rows.")
    elif impute == "mean":
        if data[missing_col].nunique() == 2:
            mode = data[missing_col].mode(dropna=True)[0]
            data[missing_col] = data[missing_col].fillna(mode)
        else:
            #print(data[missing_col].mean(skipna=True))
            mean = data[missing_col].mean(skipna=True)
            data[missing_col] = data[missing_col].fillna(mean)
    elif impute == "mice_def":
        imputer = IterativeImputer(random_state=0)
        imputer.fit(data)
        data = pd.DataFrame(imputer.transform(data), columns=data.columns)
        if data[missing_col].nunique() == 2:
            data[missing_col] = data[missing_col].round()
    elif impute == "coldel":
        return data.drop(missing_col, axis=1)
    elif impute == "knn":
        pass
    elif impute == "log_reg":
        obs_data = data.dropna()
        x = obs_data.drop(missing_col, axis = 1)
        y = obs_data[missing_col]
        model = LogisticRegression(random_state=0, max_iter=500)
        model.fit(x, y)
        #TODO fix when missing == 0
        x_miss = data[data[missing_col].isnull()].drop(missing_col,axis = 1)
        y_hat = model.predict(x_miss)
        data.loc[data[missing_col].isnull(),missing_col] = y_hat 
        
    elif len(impute_split) ==3 :
        flr = FairLogisticRegression(fairness_metric = "eo_sum",lam = int(impute_split[-1])/100)
        obs_data = data.dropna()
        
        flr.pre_fit(obs_data.drop(missing_col, axis = 1), obs_data[missing_col], epochs = 300)
        flr.fit_predicitve(obs_data.drop(response, axis = 1), obs_data[response], epochs=100)
        x = obs_data.drop(missing_col, axis = 1)
        y = obs_data[missing_col]
        y_predictive = obs_data[response]
        z = obs_data[sensitive_col]
        flr.fit(x, y, y_predictive, z, epochs = 100, 
                        data = obs_data.drop([response, missing_col], axis = 1), missing = missing_col)
        
        x_miss = data[data[missing_col].isnull()].drop(missing_col,axis = 1)
        y_hat = flr.predict(x_miss)
        data.loc[data[missing_col].isnull(),missing_col] = y_hat 
    else:
        raise NotImplementedError("NOT IMPLEMENTED THIS IMPUTATION")
        
    return data

def run(response, missing_col, sensitive, models = ["log_reg", "rf_cat", "knn"], dataset = "recid", n_runs = 10, robust = True, with_mcar = True):
    full_results = {"delta": {"mar":{metr:{m:{i:[] for i in IMPUTATION_COMBOS} for m in models} for metr in METRICS},
                              "mcar": {metr:{m:{i:[] for i in IMPUTATION_COMBOS} for m in models} for metr in METRICS}}}
    for run in tqdm(range(n_runs)):
        results = {"mar": {metr: {m: {i: [] for i in IMPUTATIONS} for m in models} for metr in METRICS},
                   "mcar": {metr: {m: {i: [] for i in IMPUTATIONS} for m in models} for metr in METRICS}}
        np.random.seed(run*13)
        if dataset =="simple":
            data = utils.load_synthetic("simple")
            alpha = SIMPLE_ALPHA
        elif dataset =="adult":
            data = utils.load_adult()
            alpha = ADULT_ALPHA
        elif dataset == "recid_synth":
            data = utils.load_synthetic()
            alpha = RECID_ALPHA
        else:
            data = utils.load_compas_alt()
            alpha = RECID_ALPHA
        
        train = data["train"]
        test = data["test"]
        class_0_test = test[test[sensitive] == 0]
        class_1_test = test[test[sensitive] == 1]
        
        noise = np.random.normal(0,0.1,size = len(train))
        
        for alph, missing_pct in zip(alpha, [0, 5,10,25,50,75]):
            #print(missing_pct)
            if with_mcar:
                data_mcar_missing = data_remover_cat(
                        train, missing_col, alph, noise, missing_pct = missing_pct, missingness="mcar", robust = robust)
            data_mar_missing = data_remover_cat(
                    train, missing_col, alph, noise, missingness="mar", robust = robust)
            #print(data_mcar_missing.columns)
            for imp in IMPUTATIONS:  # , "reg"]:
                """if imp =="fair_reg_95" or imp =="cca":
                    print(imp)"""
                #if missing_pct !=0:

                if with_mcar:
                    data_mcar = impute(data_mcar_missing, response, missing_col,sensitive_col=sensitive,alpha = alph, impute=imp)
                    #print(data_mcar.columns)
                data_mar = impute(data_mar_missing, response,  missing_col,sensitive_col=sensitive, alpha = alph, impute=imp)
                """else:
                    data_mcar = data_mcar_missing
                    data_mar = data_mar_missing"""
                for m in models:
                    if imp == "coldel" and missing_col == sensitive:
                        continue
                    #print("MODEL",m,"\n", "IMP", imp)
                    #MCAR
                    # TPR, FPR, TNR, FNR data
                    #Investigate XGBoost at the fit stage and if data needs to be transformed
                    if with_mcar:
                        predictions_0 = MODELS[m].fit(data_mcar.drop(response, axis=1), data_mcar[response]).predict(
                            class_0_test.drop([response, missing_col], axis=1) if imp == "coldel" else class_0_test.drop(response, axis=1))
                        predictions_1 = MODELS[m].fit(data_mcar.drop(response, axis=1), data_mcar[response]).predict(
                            class_1_test.drop([response, missing_col], axis=1) if imp == "coldel" else class_0_test.drop(response, axis=1))
                        
                        
                        cf_0 = utils.confusion_matrix(class_0_test[response], predictions_0)
                        cf_1 = utils.confusion_matrix(class_1_test[response], predictions_1)
                        results["mcar"]["tpr0"][m][imp].append(cf_0["Predicted true"][0])
                        results["mcar"]["tpr1"][m][imp].append(cf_1["Predicted true"][0])
                        results["mcar"]["tnr0"][m][imp].append(cf_0["Predicted false"][1])
                        results["mcar"]["tnr1"][m][imp].append(cf_1["Predicted false"][1])
                        results["mcar"]["tprd"][m][imp].append(abs(cf_1["Predicted true"][0] - cf_0["Predicted true"][0]))
                        results["mcar"]["tnrd"][m][imp].append(abs(cf_1["Predicted false"][1]-cf_0["Predicted false"][1]))
                        
                        results[m+"_mcar_"+imp+"_" +
                                str(missing_pct)+"_0"] = cf_0
                        results[m+"_mcar_"+imp+"_" +
                                str(missing_pct)+"_1"] = cf_1

                        # Fairness metrics
                        y_hat = MODELS[m].fit(data_mcar.drop(response, axis=1), data_mcar[response]).predict(
                            test.drop([response, missing_col], axis=1) if imp == "coldel" else test.drop(response, axis=1))
                        results["mcar"]["spd"][m][imp].append(
                            utils.spd(y_hat, test[sensitive]))
                        
                        results["mcar"]["acc"][m][imp].append(
                            utils.accuracy(test[response], y_hat))
                        eo = utils.equalised_odds(y_hat, test[sensitive], test[response])
                        results["mcar"]["eosum"][m][imp].append(eo["Y=1"]+eo["Y=0"])
                    # TPR, FPR, TNR, FNR data

                    #MAR
                    try:
                        predictions_0 = MODELS[m].fit(data_mar.drop(response, axis=1), data_mar[response]).predict(
                            class_0_test.drop([response, missing_col], axis=1) if imp == "coldel" else class_0_test.drop(response, axis=1))
                    except Exception as e:
                        print("Exception: ", e)
                        print("params: ", str(missing_pct)+imp)
                        print("head: ", data_mar.head())
                        print("sum: ", data_mar.sum()-len(data_mar))
                    predictions_1 = MODELS[m].fit(data_mar.drop(response, axis=1), data_mar[response]).predict(
                        class_1_test.drop([response, missing_col], axis=1) if imp == "coldel" else class_1_test.drop(response, axis=1))
                    cf_0 = utils.confusion_matrix(class_0_test[response], predictions_0)
                    cf_1 = utils.confusion_matrix(class_1_test[response], predictions_1)
                    results["mar"]["tpr0"][m][imp].append(cf_0["Predicted true"][0])
                    results["mar"]["tpr1"][m][imp].append(cf_1["Predicted true"][0])
                    results["mar"]["tnr0"][m][imp].append(cf_0["Predicted false"][1])
                    results["mar"]["tnr1"][m][imp].append(cf_1["Predicted false"][1])
                    results["mar"]["tprd"][m][imp].append(abs(cf_1["Predicted true"][0] - cf_0["Predicted true"][0]))
                    results["mar"]["tnrd"][m][imp].append(abs(cf_1["Predicted false"][1]-cf_0["Predicted false"][1]))
                    
                    results[m+"_mar_"+imp+"_" +
                            str(missing_pct)+"_0"] = cf_0
                    results[m+"_mar_"+imp+"_" +
                            str(missing_pct)+"_1"] = cf_1

                    # Fariness metrics
                    y_hat = MODELS[m].fit(data_mar.drop(response, axis=1), data_mar[response]).predict(
                        test.drop([response, missing_col], axis=1) if imp == "coldel" else test.drop(response, axis=1))
                    results["mar"]["spd"][m][imp].append(
                        utils.spd(y_hat, test[sensitive]))
                    results["mar"]["acc"][m][imp].append(
                        utils.accuracy(test[response], y_hat))

                    eo = utils.equalised_odds(y_hat, test[sensitive], test[response])
                    results["mar"]["eosum"][m][imp].append(eo["Y=1"]+eo["Y=0"])
                    #TODO add to thesis that missingness was not applied to the test data
                    #print(imp)
            """except Exception as e:
                print("EXCEPTION: ", e)"""
        
        full_results[str(run)] = results
    temp_delta = collections.defaultdict(list)
    avg = collections.defaultdict(list)
    std = {}
    
    #Collecting all observations across runs
    for miss in ["mcar", "mar"]:
        if not with_mcar and miss =="mcar":
            continue
        for key in [str(n) for n in range(n_runs)]:
            for m in models:
                for imp in IMPUTATIONS:
                    for metric in METRICS:
                        temp_delta[miss+"|"+metric+"|"+m+"|"+imp] += full_results[key][miss][metric][m][imp]
                        avg[miss+"|"+metric+"|"+m+"|"+imp].append(full_results[key][miss][metric][m][imp])
    
    #Averaging
    for key, value in avg.items():
        avg[key] = [float(n) for n in np.mean(value, axis = 0)]  
        std[key] =  [float(n) for n in np.std(value, axis = 0)] 
          
          
    try:
        save("./data/", "testresults.pickle", results)
    except Exception as e:
        print("Couldn't save data with exception: ", e)  
    try:
        save("./data/", "testaverages.pickle", avg)
    except Exception as e:
        print("Couldn't save data with exception: ", e)  
    del full_results["delta"]
    return {"Full data": full_results, "Averaged results": avg, "Standard deviation": std}