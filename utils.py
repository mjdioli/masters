from urllib import response
from fair_logistic_reg import FairLogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
import seaborn as sns

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
          "lin_reg": LinearRegression(),
          "svm": LinearSVC(random_state=0, tol=1e-5),
          "knn": KNeighborsClassifier(n_neighbors=3),
          "rf_cat": RandomForestClassifier(max_depth=4, random_state=0),
          "rf_cont": RandomForestRegressor(max_depth=4, random_state=0)}

IMPUTATIONS = ["fair_reg_99", "cca", "fair_reg_95", "mean", "mice_def", "coldel"]# , "reg"]
#IMPUTATIONS = ["cca", "mean"]#, "mice_def", "coldel"]# , "reg"]
IMPUTATION_COMBOS = [perm[0]+"|"+perm[1] for perm in permutations(IMPUTATIONS, 2)]

SAVEPATH = "experiments/"

NAME_KEYS = {"coldel": "Column deletion",
             "cca": "CCA",
             "mice_def": "Chained Eqaution",
             "mean": "Mean",
             "log_reg": "Logistic Regression",
             "svm": "SVM",
             "knn": "KNN",
             "rf_cat": "Random Forest",
             "fair_reg_99": "Fairness aware imputation lambda = 0.99",
             "fair_reg_95": "Fairness aware imputation lambda = 0.95"}

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


def sigmoid(x, alpha):
    z = np.exp(-x+alpha)
    sig = 1 / (1 + z)
    return sig


def splitter(data, response="score_factor"):
    x = data.drop(response, axis=1)
    y = data[response]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33)
    x_train[response] = y_train
    x_test[response] = y_test
    return {"train": x_train, "test": x_test}


def load_compas_alt():
    #Consider using Haewon's version, seems better
    cols = ["gender_factor", "age_factor", "race_factor",
            "priors_count", "crime_factor", "two_year_recid"] # , "score_factor"]
    new_df = pd.read_csv("./formatted_recid.csv", usecols=cols)
    new_df["is_Caucasian"] = new_df["race_factor"].apply(
        lambda x: 1 if x == "Caucasian" else 0)
    #new_df["score_factor"] = new_df["score_factor"].apply(
    #    lambda x: 0 if x == "LowScore" else 1)
    new_df["gender_factor"] = new_df["gender_factor"].apply(
        lambda x: 1 if x == "Male" else 0)
    new_df["crime_factor"] = new_df["crime_factor"].apply(
        lambda x: 0 if x == "F" else 1)
    
    #Flipping the variable so that 1 is the good outcome
    new_df["two_year_recid"] = new_df["two_year_recid"].apply(
        lambda x: 0 if x == 1 else 1)
    new_df = new_df.drop("race_factor", axis=1)
    new_df = pd.get_dummies(new_df,
                            columns=["age_factor"],
                            drop_first=True)
    return splitter(new_df, response = "two_year_recid")


def compas_cleaning(df):
    new_df = df.dropna()
    new_df = new_df[(new_df["days_b_screening_arrest"] <= 30) &
                    (new_df["days_b_screening_arrest"] >= -30) &
                    (new_df["is_recid"] != -1) &
                    (new_df["c_charge_degree"] != "O") &
                    (new_df["score_text"] != "N/A")]
    new_df["length_of_stay"] = (
        (pd.to_datetime(df["c_jail_out"])-pd.to_datetime(df["c_jail_in"])).dt.days)
    new_df["length_of_stay"] = new_df["length_of_stay"].astype(int)

    #Perhaps limit dataset to only black and white participants
    new_df["is_Caucasian"] = new_df["race"].apply(
        lambda x: 1 if x == "Caucasian" else 0)
    new_df.drop(labels=["c_jail_out", "c_jail_in",
                "days_b_screening_arrest", "is_recid", "race"], axis=1, inplace=True)
    if "v_score_text" in new_df.columns:
        new_df.columns = ["score_text" if col ==
                          "v_score_text" else col for col in new_df.columns]
    new_df["score_text"] = new_df["score_text"].apply(
        lambda x: 0 if x == "High" else 1)
    new_df = pd.get_dummies(new_df,
                            columns=["c_charge_degree",
                                     "age_cat",
                                     "sex"],
                            drop_first=True)
    col_names = new_df.columns
    for i in range(len(col_names)):
        if col_names[i] == "age_factor_Greater than 45":
            col_names[i] = "age_factor_greater_than_45"
        elif col_names[i] == "age_factor_Less than 25":
            col_names[i] = "age_factor_less_than_25"
    return new_df

import sklearn.preprocessing as preprocessing
from collections import namedtuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def balance_data(df, attr, major=1): 
    """Borrowed from https://github.com/haewon55/FairMIPForest
    """
    sample_size = len(df[df[attr] == major]) - len(df[df[attr]==(1-major)])
    if sample_size < 0:
        raise ValueError
    
    np.random.seed(0)
    drop_idx = np.random.choice(df[df[attr]==major].index, sample_size, replace=False)
    return df.drop(drop_idx)


def load_adult(smaller=False, scaler=True):
    '''
    Borrowed from https://github.com/haewon55/FairMIPForest
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.
    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    pwd = 'data/'
    data = pd.read_csv(
        Path(pwd+'adult.data'),
        names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"]
            )
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(
        pwd+'adult.test',
        names=["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"]
    )
    ordering = ["age", "fnlwgt", "education", "education-num",
            "occupation", "relationship",  "capital-gain", "capital-loss",
            "hours-per-week", "native-country","race", "gender","marital-status","workclass", "income"]
    data = pd.concat([data, data_test])
    data = data[ordering]
    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = data["workclass"][4].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    #print("FIRST",data.head(2))
    #Convert race to simply white or non-white
    data["race"] = data["race"].apply(lambda x: 0 if x==" White" else 1)
    data["workclass"] = data["workclass"].apply(lambda x: 0 if "gov" in x else 1)

    data["marital-status"] = data["marital-status"].apply(lambda x: 1 if "arried" in x else 0)
    # categorical fields
    category_col = ['education',  'occupation',
                    'relationship', 'gender', 'native-country', 'income']

    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c

    
    
    data, data_test = data.drop(columns=['fnlwgt']), data_test.drop(columns=['fnlwgt'])
    
    datamat = data.values
    datamat = datamat[:, :-5]
    
    if scaler:
        scaler = MinMaxScaler()
        scaler.fit(datamat)
        data.iloc[:, :-5] = scaler.fit_transform(datamat)
#         data.iloc[:, -5] = target


    #return {"train": data.iloc[:len_train], "test": data.iloc[len_train:]}
    return splitter(data, response="income")


def load_compas(version="standard"):
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
    violent_recid = pd.read_csv(
        "./compas_violent_recid.csv", usecols=cols+["v_score_text"])
    if version == "standard":
        return splitter(compas_cleaning(recid))
    else:
        return splitter(compas_cleaning(violent_recid))


def load_synthetic(ver="recid_alt"):
    if ver == "recid":
        size = 6000
        priors_count = np.round(
            np.abs(np.random.uniform(3.246, 4.743, size=size)))
        two_year_recid = np.random.binomial(1, 0.455, size=size)
        is_Caucasian = np.random.binomial(1, 0.34, size=size)
        crime_factor = np.random.binomial(1, 0.3567, size=size)
        age_greater_than_45 = np.random.binomial(1, 0.209, size=size)
        age_less_than_25 = np.random.binomial(1, 0.218, size=size)
        sex_male = np.random.binomial(1, 0.809, size=size)
        score_text = np.around(sigmoid((priors_count*(0.1)+two_year_recid+is_Caucasian+crime_factor +
                                        age_greater_than_45+age_less_than_25+sex_male), alpha=3)).astype(int)
        synth_cat = pd.DataFrame({"score_factor": score_text, "priors_count": priors_count,
                                  "two_year_recid": two_year_recid, "is_Caucasian": is_Caucasian, "crime_factor": crime_factor,
                                  "age_factor_greater_than_45": age_greater_than_45, "age_factor_less_than_25": age_less_than_25, "gender_factor": sex_male})
        synth_cat_test = synth_cat.iloc[:round(0.333*size), :]
        synth_cat_train = synth_cat.iloc[round(0.333*size):, :]
    elif ver == "recid_alt":
        size = 6000
        priors_count = np.round(
            np.abs(np.random.uniform(3.246, 4.743, size=size)))
        #two_year_recid = np.random.binomial(1, (1-0.455), size=size)
        is_Caucasian = np.random.binomial(1, 0.34, size=size)
        crime_factor = np.random.binomial(1, 0.3567, size=size)
        age_greater_than_45 = np.random.binomial(1, 0.209, size=size)
        age_less_than_25 = np.random.binomial(1, 0.218, size=size)
        sex_male = np.random.binomial(1, 0.809, size=size)
        two_year_recid = np.around(sigmoid((priors_count*(0.2)+is_Caucasian+crime_factor +
                                age_greater_than_45+age_less_than_25+sex_male), alpha=2.6)).astype(int)
        synth_cat = pd.DataFrame({"priors_count": priors_count, "two_year_recid": two_year_recid,
                                  "is_Caucasian": is_Caucasian, "crime_factor": crime_factor,
                                  "age_factor_greater_than_45": age_greater_than_45, 
                                  "age_factor_less_than_25": age_less_than_25, "gender_factor": sex_male})
        synth_cat_test = synth_cat.iloc[:round(0.333*size), :]
        synth_cat_train = synth_cat.iloc[round(0.333*size):, :]
    elif ver == "simple":
        size = 10000
        x_1 = np.random.binomial(1, 0.45, size=size)
        x_2 = np.random.binomial(1, 0.65, size=size)
        x_3 = np.random.normal(0,1,size)
        y = np.around(sigmoid(x_1*0.3+x_2+x_3, alpha=0.8)).astype(int)
        synth_cat = pd.DataFrame({"y": y, "x_1": x_1, "x_2": x_2, "x_3": x_3})
        synth_cat_test = synth_cat.iloc[:round(0.333*size), :]
        synth_cat_train = synth_cat.iloc[round(0.333*size):, :]
    else:
        #TODO change parameters for violent
        size = 4000
        priors_count = np.round(
            np.abs(np.random.uniform(3.246, 4.743, size=size)))
        two_year_recid = np.random.binomial(1, 0.455, size=size)
        is_Caucasian = np.random.binomial(1, 0.34, size=size)
        crime_factor = np.random.binomial(1, 0.3567, size=size)
        age_greater_than_45 = np.random.binomial(1, 0.209, size=size)
        age_less_than_25 = np.random.binomial(1, 0.218, size=size)
        sex_male = np.random.binomial(1, 0.809, size=size)
        score_text = np.around(sigmoid((priors_count*(0.1)+two_year_recid+is_Caucasian+crime_factor +
                                        age_greater_than_45+age_less_than_25+sex_male), alpha=3)).astype(int)
        synth_cat = pd.DataFrame({"score_factor": score_text, "priors_count": priors_count,
                                  "two_year_recid": two_year_recid, "is_Caucasian": is_Caucasian, "crime_factor": crime_factor,
                                  "age_factor_greater_than_45": age_greater_than_45, "age_factor_less_than_25": age_less_than_25, "gender_factor": sex_male})
        synth_cat_test = synth_cat.iloc[:round(0.333*size), :]
        synth_cat_train = synth_cat.iloc[round(0.333*size):, :]

    return {"test": synth_cat_test, "train": synth_cat_train}

#Note just changed positive to True


def spd(pred, protected_class, positive=True):
    """
    Equation: |P(Y_pred = y | Z = 1) - P(Y_pred = y | Z = 0)|
    Assumes that the positive class is the desired outcome and
        that the protected_class is 0/1 binary"""
    z_1 = [y_hat for y_hat, z in zip(
        pred, np.array(protected_class)) if z == 1]
    z_0 = [y_hat for y_hat, z in zip(
        pred, np.array(protected_class)) if z == 0]

    if not positive:
        z_1 = [0 if z == 1 else 1 for z in z_1]
        z_0 = [0 if z == 1 else 1 for z in z_1]
    """if len(z_1)+len(z_0)!=len(pred):
        print("NOT EQUAL")"""
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
    
def eo_sum(pred, prot, true):
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
    return abs(sum(z1_y1)/len(z1_y1)-sum(z0_y1)/len(z0_y1)) + abs(sum(z1_y0)/len(z1_y0)-sum(z0_y0)/len(z0_y0))
    


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
    except:
        print("PP error z1")
        z1 = 0

    try:
        z0 = sum(z0_yhat1)/len(z0_yhat1)
    except:
        z0 = 0
        print("PP error z0")

    if z1 == 0 and z0 == 0:
        print("Both PP values are zero!")

    return abs(z1-z0)


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


def accuracy(true, pred):
    correct = [1 if t == p else 0 for t, p in zip(true, pred)]
    if len(correct) > 0:
        return sum(correct)/len(correct)
    else:
        return 0


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
        data["missing"] = [1 if m == 1 else 0 for m in mcar]
        data[missing_col] = data[missing_col].mask(data["missing"] == 1,
                                                   other=np.nan)
        data = data.drop("missing", axis=1)
        #print("MCAR")
    return data


#TODO unfinished
def regression_imputer(full_data, missing):
    # Check if data is binary categorical or continuous.
    cca = full_data.dropna()
    if cca[missing].nunique() == 2:
        pass
    else:
        pass

#TODO add column deletion "imputation"


@ignore_warnings(category=ConvergenceWarning)
def impute(dataframe, missing_col,sensitive_col, impute="cca"):
    #TODO add knn imputation
    data = dataframe.copy()
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
    elif impute == "reg":
        if data[missing_col].nunique() == 2:
            model = LogisticRegression(random_state=0, max_iter=300)
        else:
            model = LinearRegression()
        model = model.fit(data.dropna().drop(
            missing_col, axis=1), data.dropna()[missing_col])
        data = data.fillna(-1)
        for i, row in data.iterrows():
            if row[missing_col] == -1:
                data.loc[i, missing_col] = model.predict(
                    row.drop(missing_col))[0]
        print("NANS: ", data.value_counts())
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
    elif impute == "coldel":
        return data.drop(missing_col, axis=1)
    elif impute == "knn":
        pass
    elif impute=="fair_reg_95":
        flr = FairLogisticRegression(fairness_metric = "eo_sum",lam = 0.95)
        obs_data = data.dropna()
        x = obs_data.drop(missing_col, axis = 1)
        y = obs_data[missing_col]
        z = obs_data[sensitive_col]
        flr.fit(x,y,z, epochs=50)
        
        x_miss = data[data[missing_col].isnull()].drop(missing_col,axis = 1)
        y_hat = flr.predict(x_miss)
        data.loc[data[missing_col].isnull(),missing_col] = y_hat 
        
    elif impute =="fair_reg_99":
        flr = FairLogisticRegression(fairness_metric = "eo_sum",lam = 0.99)
        obs_data = data.dropna()
        x = obs_data.drop(missing_col, axis = 1)
        y = obs_data[missing_col]
        z = obs_data[sensitive_col]
        flr.fit(x,y,z, epochs=50)
        
        x_miss = data[data[missing_col].isnull()].drop(missing_col,axis = 1)
        y_hat = flr.predict(x_miss)
        data.loc[data[missing_col].isnull(),missing_col] = y_hat 
    return data


def load_data(dataset="compas"):
    #Loading helper function
    if dataset == "compas":
        return load_compas_alt()
    elif dataset == "synthetic":
        return load_synthetic()
    elif dataset == "adult":
        return load_adult(scaler = True)
    elif dataset == "simple":
        return load_synthetic(ver = "simple")
    else:
        raise AttributeError("Wrong dataset name")

#TODO Note down overall accuracy scores in a json.
#TODO get on the above
#TODO get average TPR, TNR, EO, PP, SPD across all models and plot


@ignore_warnings(category=ConvergenceWarning)
def test_bench(pred: str, missing: str, sensitive: str, data="compas", pred_var_type: str = "cat",
               percentiles=None, n_runs=1, differencing = True):
    if pred_var_type == "cat":
        models = ["log_reg", "rf_cat", "svm", "knn"]
    else:
        models = ["lin_reg", "rf_cont", "knn"]
    # print("class_0",sum(class_0_test[pred]))
    # print("class_1",sum(class_1_test[pred]))
    metrics = ["spd", "eo0","eo1", "eosum", "pp", "acc", "tpr0", "tpr1", "tnr0", "tnr1"]
    #delta = {metr:{m:{i:[] for i in IMPUTATIONS} for m in models} for metr in metrics}
    full_results = {"delta": {"mar":{metr:{m:{i:[] for i in IMPUTATION_COMBOS} for m in models} for metr in metrics},
                              "mcar": {metr:{m:{i:[] for i in IMPUTATION_COMBOS} for m in models} for metr in metrics}}}
    
    if percentiles is None:
        percentiles = [i for i in range(
            1, 16)]+[j for j in range(20, 100, 10)]
    for i in tqdm(range(n_runs)):
        loaded_data = load_data(data)
        train = loaded_data["train"]
        test = loaded_data["test"]

        # sensitive var
        class_0_test = test[test[sensitive] == 0]
        class_1_test = test[test[sensitive] == 1]
        
        results = {"mar": {metr: {m: {i: [] for i in IMPUTATIONS} for m in models} for metr in metrics},
                   "mcar": {metr: {m: {i: [] for i in IMPUTATIONS} for m in models} for metr in metrics}}
        #results["mar"]["delta"] = {metr: {m: {i: [] for i in IMPUTATION_COMBOS} for m in models} for metr in metrics}
        #results["mcar"]["delta"] = {metr: {m: {i: [] for i in IMPUTATION_COMBOS} for m in models} for metr in metrics}
        #print("results", results)
        for p in percentiles:   
            data_mcar_missing = data_remover_cat(
                train, missing, p, missing="mcar")
            data_mar_missing = data_remover_cat(
                train, missing, p, missing="mar")
            for imp in IMPUTATIONS:  # , "reg"]:
                data_mcar = impute(data_mcar_missing, missing,sensitive_col=sensitive, impute=imp)
                data_mar = impute(data_mar_missing, missing,sensitive_col=sensitive, impute=imp)
                for m in models:
                    if imp == "coldel" and missing == sensitive:
                        continue

                    #MCAR
                    # TPR, FPR, TNR, FNR data
                    predictions_0 = MODELS[m].fit(data_mcar.drop(pred, axis=1), data_mcar[pred]).predict(
                        class_0_test.drop([pred, missing], axis=1) if imp == "coldel" else class_0_test.drop(pred, axis=1))
                    predictions_1 = MODELS[m].fit(data_mcar.drop(pred, axis=1), data_mcar[pred]).predict(
                        class_1_test.drop([pred, missing], axis=1) if imp == "coldel" else class_0_test.drop(pred, axis=1))
                    
                    
                    cf_0 = confusion_matrix(class_0_test[pred], predictions_0)
                    cf_1 = confusion_matrix(class_1_test[pred], predictions_1)
                    results["mcar"]["tpr0"][m][imp].append(cf_0["Predicted true"][0])
                    results["mcar"]["tpr1"][m][imp].append(cf_1["Predicted true"][0])
                    results["mcar"]["tnr0"][m][imp].append(cf_0["Predicted false"][0])
                    results["mcar"]["tnr1"][m][imp].append(cf_1["Predicted false"][0])
                    
                    results[m+"_mcar_"+imp+"_" +
                            str(p)+"_0"] = cf_0
                    results[m+"_mcar_"+imp+"_" +
                            str(p)+"_1"] = cf_1

                    # Fairness metrics
                    y_hat = MODELS[m].fit(data_mcar.drop(pred, axis=1), data_mcar[pred]).predict(
                        test.drop([pred, missing], axis=1) if imp == "coldel" else test.drop(pred, axis=1))
                    results["mcar"]["spd"][m][imp].append(
                        spd(y_hat, test[sensitive]))
                    
                    results["mcar"]["pp"][m][imp].append(
                        predictive_parity(y_hat, test[sensitive], test[pred]))
                    results["mcar"]["acc"][m][imp].append(
                        accuracy(test[pred], y_hat))
                    eo = equalised_odds(y_hat, test[sensitive], test[pred])
                    results["mcar"]["eo0"][m][imp].append(eo["Y=0"])
                    results["mcar"]["eo1"][m][imp].append(eo["Y=1"])
                    results["mcar"]["eosum"][m][imp].append(eo["Y=1"]+eo["Y=0"])
                    # TPR, FPR, TNR, FNR data

                    #MAR
                    try:
                        predictions_0 = MODELS[m].fit(data_mar.drop(pred, axis=1), data_mar[pred]).predict(
                            class_0_test.drop([pred, missing], axis=1) if imp == "coldel" else class_0_test.drop(pred, axis=1))
                    except Exception as e:
                        print("Exception: ", e)
                        print("params: ", str(p)+imp)
                        print("head: ", data_mar.head())
                        print("sum: ", data_mar.sum()-len(data_mar))
                    predictions_1 = MODELS[m].fit(data_mar.drop(pred, axis=1), data_mar[pred]).predict(
                        class_1_test.drop([pred, missing], axis=1) if imp == "coldel" else class_1_test.drop(pred, axis=1))
                    cf_0 = confusion_matrix(class_0_test[pred], predictions_0)
                    cf_1 = confusion_matrix(class_1_test[pred], predictions_1)
                    results["mar"]["tpr0"][m][imp].append(cf_0["Predicted true"][0])
                    results["mar"]["tpr1"][m][imp].append(cf_1["Predicted true"][0])
                    results["mar"]["tnr0"][m][imp].append(cf_0["Predicted false"][0])
                    results["mar"]["tnr1"][m][imp].append(cf_1["Predicted false"][0])
                    
                    results[m+"_mar_"+imp+"_" +
                            str(p)+"_0"] = cf_0
                    results[m+"_mar_"+imp+"_" +
                            str(p)+"_1"] = cf_1

                    # Fariness metrics
                    y_hat = MODELS[m].fit(data_mar.drop(pred, axis=1), data_mar[pred]).predict(
                        test.drop([pred, missing], axis=1) if imp == "coldel" else test.drop(pred, axis=1))
                    results["mar"]["spd"][m][imp].append(
                        spd(y_hat, test[sensitive]))
                    results["mar"]["pp"][m][imp].append(
                        predictive_parity(y_hat, test[sensitive], test[pred]))
                    results["mar"]["acc"][m][imp].append(
                        accuracy(test[pred], y_hat))

                    eo = equalised_odds(y_hat, test[sensitive], test[pred])
                    results["mar"]["eo0"][m][imp].append(eo["Y=0"])
                    results["mar"]["eo1"][m][imp].append(eo["Y=1"])
                    results["mar"]["eosum"][m][imp].append(eo["Y=1"]+eo["Y=0"])
                    #TODO add to thesis that missingness was not applied to the test data
                    #print(imp)

                    #Only seems to be an issue with synthetic data as performance is consistently high
                    """if prev_spd == np.sum(y_hat):
                        print("UNCHANGED SPD!", prev_spd, len(data_mar), imp)
                    else:
                        prev_spd = np.sum(y_hat)
                    if len(results["mar"]["spd"][m][imp]) ==0:
                        print("SPD LEN = 0! Y_HAT = ", y_hat)"""
        #full_results[str(i)] = results.copy()
        #full_results[str(i)] = copy.deepcopy(results)
        
        full_results[str(i)] = results
    temp_delta = collections.defaultdict(list)
    avg = collections.defaultdict(list)
    
    #Collecting all observations across runs
    for miss in ["mcar", "mar"]:
        for key in [str(n) for n in range(n_runs)]:
            for m in models:
                for imp in IMPUTATIONS:
                    for metric in metrics:
                        temp_delta[miss+"|"+metric+"|"+m+"|"+imp] += full_results[key][miss][metric][m][imp]
                        avg[miss+"|"+metric+"|"+m+"|"+imp].append(full_results[key][miss][metric][m][imp])
    
    #Averaging
    for key, value in avg.items():
        avg[key] = [float(n) for n in np.mean(value, axis = 0)]  
          
          
    try:
        save("./data/", "testresults.pickle", results)
    except Exception as e:
        print("Couldn't save data with exception: ", e)  
    if differencing:        
        imp_keys = []
        metric_keys = []
        for key1, value1 in temp_delta.items():
            for key2, value2 in temp_delta.items():
                if key1==key2:
                    continue
                keys1 = key1.split("|")
                miss1, metric1, m1, imp1 =keys1[0],keys1[1],keys1[2],keys1[3]
                keys2 = key2.split("|")
                miss2, metric2, m2, imp2 =keys2[0],keys2[1],keys2[2],keys2[3]
                if (miss1!=miss2) or (metric1!=metric2) or (m1!=m2) or (imp1==imp2):
                    continue
                
                if ((imp1+"|"+imp2 in imp_keys) or (imp2+"|"+imp1 in imp_keys)) and metric1 in metric_keys:
                    continue
                
                imp_keys.append(imp2+"|"+imp1)
                imp_keys.append(imp1+"|"+imp2)
                metric_keys.append(metric1)
                
                #TODO FIX HERE
                #done_keys.append(miss1+metric1+m1+imp1+"|"+imp2)
                temp_differences = []
                for v1 in value1:
                    temp_differences+=[v1-v2 for v2 in value2]
                full_results["delta"][miss1][metric1][m1][imp1+"|"+imp2] = temp_differences
    

    #Plotting differencings
    
        key = data
        if not os.path.isdir(Path(SAVEPATH)):
            os.mkdir(Path(SAVEPATH))
        savepath = SAVEPATH+missing+"_"+sensitive+"_"+key+"/"
        if not os.path.isdir(Path(savepath)):
            os.mkdir(Path(savepath))
        if not os.path.isdir(Path(savepath+"/differencing/")):
            os.mkdir(Path(savepath+"/differencing/"))
        savepath = savepath+"differencing/" #TODO investigate if savepath is correct
        #a = differencing_models(full_results, "mar","acc","log_reg","cca")
        #b = differencing_models(full_results, "mcar","acc","log_reg","cca")
        #print(a==b)
        #print("a: ", a, "\n", "b: ", b)
        #print(metrics)
        for miss in ["mcar", "mar"]:
            for m in models:
                for metric in metrics:
                    for imp1 in IMPUTATIONS:
                        for imp2 in IMPUTATIONS:
                            if imp1==imp2 or (imp1+"|"+imp2 not in full_results["delta"][miss][metric][m]) or (imp2+"|"+imp1 not in full_results["delta"][miss][metric][m]):
                                continue
                            #TODO fix differencing step
                            #print("Missingness", miss, "IMP", imp1+"|"+imp2, "METRIC", metric)
                            if full_results["delta"]["mar"][metric][m][imp1+"|"+imp2] == full_results["delta"]["mar"][metric][m][imp1+"|"+imp2]:
                                #print("SAME")
                                pass
                            plotting_differencing(
                                bucketiser(
                                    differencing_models(full_results, miss,metric,m,imp1+"|"+imp2)
                                    ,0.3),
                                title = "Differencing of " + NAME_KEYS[m] + " with " + NAME_KEYS[imp1] + "|"+ NAME_KEYS[imp2] + " measured by " + metric,
                                savepath= savepath+miss+"_"+m+"_"+imp1+"_"+imp2+"_"+metric+".png")
        
        #Deleting differencings after they are no longer needed
    del full_results["delta"]
    return {"Full data": full_results, "Averaged results": avg}

def bucketiser(count_dict, max):
    lin =  np.linspace(-max, max, 30)
    buckets = {str(v):0 for v in lin}
    for key, value in count_dict.items():
        prev = False
        next = False
        for l in lin:
            if prev and next:
                buckets[str(l)] = buckets[str(l)] + value
                break
            elif prev:
                if float(key)<l:
                    next = True
            elif float(key)>l:
                prev = True
    return buckets

def differencing_models(results, missing, metric, model, imputation):
    test = {}
    #print("DIFFERENCES", results["delta"][missing][metric][model][imputation])
    #print("IMPUTATION", imputation)
    for val in results["delta"][missing][metric][model][imputation]:
        if val not in test:
            test[val] = 1
        else:
            test[val] = test[val]+1
    return test

def plotting_differencing(buckets, title, savepath):
    if len(buckets) == 0:
        print("LENGTH IS 0")
        pass
    else:
        #print("SAVING IN", savepath)
        sns.set_theme(style="whitegrid")
        fig = plt.gcf()
        fig.set_size_inches(20, 11)
        sns.barplot(x = [round(float(a), 4) for a in list(buckets.keys())], y = list(buckets.values()))
        plt.title(title)
        plt.savefig(Path(savepath))
        #plt.show()
        plt.clf()




#TODO REDO PLOTTING TO SUPPORT THE NEW DATA FORMAT
#TODO Add filename thingy
def plotting_cf(models, correctives, results, key=None):
    table = {}
    if key is None:
        if not os.path.isdir(Path(SAVEPATH)):
            os.mkdir(Path(SAVEPATH))
        savepath = SAVEPATH
    else:
        if not os.path.isdir(Path(SAVEPATH+key)):
            os.mkdir(Path(SAVEPATH+key))
        savepath = SAVEPATH+key
    averages = {"mcar": {c: {"tpr": {"0": [], "1": []}, "tnr": {"0": [], "1": []}} for c in correctives},
                "mar": {c: {"tpr": {"0": [], "1": []}, "tnr": {"0": [], "1": []}} for c in correctives}}
    for m in models:
        for c in correctives:
            tpr_mar = {"0": {}, "1": {}}
            tnr_mar = {"0": {}, "1": {}}
            tpr_mcar = {"0": {}, "1": {}}
            tnr_mcar = {"0": {}, "1": {}}
            for k, value in results.items():
                if m+"_mar" in k and c in k:
                    if k[-1] == "0":
                        tpr_mar["0"][int(k.split("_")[-2])
                                     ] = value["Predicted true"][0]
                        tnr_mar["0"][int(k.split("_")[-2])
                                     ] = value["Predicted false"][1]
                    else:
                        tpr_mar["1"][int(k.split("_")[-2])
                                     ] = value["Predicted true"][0]
                        tnr_mar["1"][int(k.split("_")[-2])
                                     ] = value["Predicted false"][1]
                elif m+"_mcar" in k and c in k:
                    try:
                        if k[-1] == "0":
                            tpr_mcar["0"][int(k.split("_")[-2])
                                          ] = value["Predicted true"][0]
                            tnr_mcar["0"][int(k.split("_")[-2])
                                          ] = value["Predicted false"][1]
                        else:
                            tpr_mcar["1"][int(k.split("_")[-2])
                                          ] = value["Predicted true"][0]
                            tnr_mcar["1"][int(k.split("_")[-2])
                                          ] = value["Predicted false"][1]
                    except Exception as e:
                        print("k", k, "exception", e)
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
            #plt.ylim(0.6, 1.01)
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
            #plt.ylim(0.6, 1.01)
            plt.legend()
            plt.savefig(Path(savepath+m+"_"+c+"_MCAR.png"))
            plt.clf()

            averages["mar"][c]["tpr"]["0"].append(list(tpr_mar["0"].values()))
            averages["mar"][c]["tpr"]["1"].append(list(tpr_mar["1"].values()))
            averages["mar"][c]["tnr"]["0"].append(list(tnr_mar["0"].values()))
            averages["mar"][c]["tnr"]["1"].append(list(tnr_mar["1"].values()))
            averages["mcar"][c]["tpr"]["0"].append(
                list(tpr_mcar["0"].values()))
            averages["mcar"][c]["tpr"]["1"].append(
                list(tpr_mcar["1"].values()))
            averages["mcar"][c]["tnr"]["0"].append(
                list(tnr_mcar["0"].values()))
            averages["mcar"][c]["tnr"]["1"].append(
                list(tnr_mcar["1"].values()))
            #print(tpr_mcar)
            #print(list(tpr_mar["0"].values())[10])
            table[m+"_"+c+"_MCAR"] = [float(list(tpr_mcar["1"].values())[10]), float(list(tpr_mcar["0"].values())[10]),
                                      (list(tnr_mcar["1"].values())[10]), float(list(tnr_mcar["0"].values())[10])]
            table[m+"_"+c+"_MAR"] = [float(list(tpr_mar["1"].values())[10]), float(list(tpr_mar["0"].values())[10]),
                                     float(list(tnr_mar["1"].values())[10]), float(list(tnr_mar["0"].values())[10])]
    #print(table)

    for c in correctives:
        plt.plot(list(tpr_mcar["0"].keys()), np.mean(averages["mcar"][c]["tpr"]["0"], axis=0),
                 label="TPR MCAR class 0")
        plt.plot(list(tpr_mcar["1"].keys()), np.mean(averages["mcar"][c]["tpr"]["1"], axis=0),
                 label="TPR MCAR class 1")
        plt.plot(list(tnr_mcar["0"].keys()), np.mean(averages["mcar"][c]["tnr"]["0"], axis=0),
                 label="TNR MCAR class 0")
        plt.plot(list(tnr_mcar["1"].keys()), np.mean(averages["mcar"][c]["tnr"]["1"], axis=0),
                 label="TNR MCAR class 1")
        plt.title("Average performance of " + c + " across models")
        plt.xlabel("Missingness percent")
        plt.ylabel("Accuracy")
        #plt.ylim(0.6, 1.01)
        plt.legend()
        plt.savefig(Path(savepath+"average_"+c+"_MCAR.png"))
        plt.clf()

        plt.plot(list(tpr_mcar["0"].keys()), np.mean(averages["mar"][c]["tpr"]["0"], axis=0),
                 label="TPR MAR class 0")
        plt.plot(list(tpr_mcar["1"].keys()), np.mean(averages["mar"][c]["tpr"]["1"], axis=0),
                 label="TPR MAR class 1")
        plt.plot(list(tnr_mcar["0"].keys()), np.mean(averages["mar"][c]["tnr"]["0"], axis=0),
                 label="TNR MAR class 0")
        plt.plot(list(tnr_mcar["1"].keys()), np.mean(averages["mar"][c]["tnr"]["1"], axis=0),
                 label="TNR MAR class 1")
        plt.title("Average performance of " + c + " across models")
        plt.xlabel("Missingness percent")
        plt.ylabel("Accuracy")
        #plt.ylim(0.6, 1.01)
        plt.legend()
        plt.savefig(Path(savepath+"average_"+c+"_MAR.png"))
        plt.clf()
    return_df = pd.DataFrame(
        table, index=["TPR Z=1", "TPR Z=0", "TNR Z=1", "TNR Z=0"])
    #print(return_df.iloc[:,0])
    save(savepath, "cf_table_data_"+key.split("/")[0]+".json", table)
    return return_df


def plotting_others(results, key=None):
    if key is None:
        if not os.path.isdir(Path(SAVEPATH)):
            os.mkdir(Path(SAVEPATH))
        savepath = SAVEPATH
    else:
        if not os.path.isdir(Path(SAVEPATH+key)):
            os.mkdir(Path(SAVEPATH+key))
        savepath = SAVEPATH+key
    table = {}
    for missingness, data in results.items():
        if missingness != "mar" and missingness != "mcar":
            continue
        for metric, res in data.items():
            for model, imps in res.items():
                for imputation, vals in imps.items():
                    table[model+"_"+imputation+"_"+missingness] = {}
                    if len(vals) == 0:
                        continue
                    if isinstance(vals[0], dict):
                        y_0 = np.zeros(len(vals))
                        y_1 = np.zeros(len(vals))

                        for i, dictio in enumerate(vals):
                            y_0[i] = dictio["Y=0"]
                            y_1[i] = dictio["Y=1"]

                        fig = plt.gcf()
                        fig.set_size_inches(18.5, 10.5)
                        plt.plot(list(results["percentiles"]), list(
                            y_0), label=model+"_"+imputation+"Y=0")
                        plt.plot(list(results["percentiles"]), list(
                            y_1), label=model+"_"+imputation+"Y=1")
                        plt.title(model+"_"+"_"+metric+"_"+missingness)
                        plt.xlabel("Missingness percent")
                        plt.ylabel(metric)
                        plt.legend()
                        table[model+"_"+imputation+"_" +
                              missingness][metric+"_Y=0"] = y_0[10]
                        table[model+"_"+imputation+"_" +
                              missingness][metric+"_Y=1"] = y_1[10]

                    else:
                        fig = plt.gcf()
                        fig.set_size_inches(18.5, 10.5)
                        plt.plot(list(results["percentiles"]), list(
                            vals), label=model+"_"+imputation)
                        plt.title(model+"_"+"_"+metric+"_"+missingness)
                        plt.xlabel("Missingness percent")
                        plt.ylabel(metric)
                        plt.legend()
                        table[model+"_"+imputation+"_" +
                              missingness][metric] = vals[10]
                plt.savefig(Path(savepath+model+"_"+imputation +
                            "_"+metric+"_"+missingness+".png"))
                plt.clf()
    print(table)
    save(savepath, "metrics_table_data_"+key.split("/")[0]+".json", table)
    return pd.DataFrame(table, index=["spd", "pp", "eo_Y=0", "eo_Y=1"])


def averaging_results(results):
    averaged = {}
    #model parameters
    for k1, v1 in results.items():
        #A given iteration
        for k2, v2 in v1.items():
            for k3, v3 in v2.items():
                if isinstance(v3, list):
                    full_list = []
                    for i in range(len(results)):
                        full_list.append(results[k1][str(i)][k3])
                    averaged[k1+"|"+k3] = {"mean": np.mean(full_list, axis=0),
                                           "std": np.std(full_list, axis=0)}
                    continue
                for k4, v4 in v3.items():
                    if isinstance(v4, list):
                        #print(k1+"|"+k3+"|"+k4+"|"+k5)
                        full_list = []
                        for i in range(len(results)):
                            full_list.append(results[k1][str(i)][k3][k4])
                        averaged[k1+"|"+k3+"|"+k4] = {"mean": np.mean(full_list, axis=0),
                                                      "std": np.std(full_list, axis=0)}
                        continue
                    for k5, v5 in v4.items():
                        if isinstance(v5, list):
                            #print(k1+"|"+k3+"|"+k4+"|"+k5)
                            full_list = []
                            for i in range(len(results)):
                                full_list.append(
                                    results[k1][str(i)][k3][k4][k5])
                            averaged[k1+"|"+k3+"|"+k4+"|"+k5] = {"mean": np.mean(full_list, axis=0),
                                                                 "std": np.std(full_list, axis=0)}
                            continue
                        for k6, v6 in v5.items():
                            if isinstance(v6, list):
                                try:
                                    if isinstance(v6[0], dict):
                                        pass
                                    else:
                                        full_list = []
                                        for i in range(len(results)):
                                            full_list.append(
                                                results[k1][str(i)][k3][k4][k5][k6])
                                        #print(k1+"|"+k3+"|"+k4+"|"+k5+"|"+k6)
                                        averaged[k1+"|"+k3+"|"+k4+"|"+k5+"|"+k6] = {"mean": np.mean(full_list, axis=0),
                                                                                    "std": np.std(full_list, axis=0)}
                                except:

                                    print(k1+"|"+k3+"|"+k4+"|" +
                                          k5+"|"+k6, "\n", v6)
                                continue
                            """for k7, v7 in v6.items():
                                try:
                                    if isinstance(v7[0], dict):
                                        pass
                                    else:
                                        full_list = []
                                        for i in range(len(results)):
                                            full_list.append(results[k1][str(i)][k3][k4][k5][k6][k7])
                                        #print(k1+"|"+k3+"|"+k4+"|"+k5+"|"+k6)
                                        averaged[k1+"|"+k3+"|"+k4+"|"+k5+"|"+k6+"|"+k7] = {"mean": np.mean(full_list, axis=0),
                                                                "std": np.std(full_list, axis=0)}
                                except:
                                    
                                    print(k1+"|"+k3+"|"+k4+"|"+k5+"|"+k6+"|"+k7, "\n", v7)
                                continue"""
    return averaged


def dict_initialiser(data, experiment, model, percentiles):
    p = {str(p): -1 for p in percentiles}

    metric_keys = [i+"|"+"spd" for i in IMPUTATIONS]
    metric_keys += [i+"|"+"eo" for i in IMPUTATIONS]
    metric_keys += [i+"|"+"pp" for i in IMPUTATIONS]
    metric_keys += [i+"|"+"acc" for i in IMPUTATIONS]
    data[experiment][model] = {"mar": {m: [-1]*len(p) for m in metric_keys}, "mcar": {m: [-1]*len(p) for m in metric_keys},
                               "tpr1|mar": {imp: p.copy() for imp in IMPUTATIONS}, "tpr0|mar": {imp: p.copy() for imp in IMPUTATIONS},
                               "tnr1|mar": {imp: p.copy() for imp in IMPUTATIONS}, "tnr0|mar": {imp: p.copy() for imp in IMPUTATIONS},
                               "tpr1|mcar": {imp: p.copy() for imp in IMPUTATIONS}, "tpr0|mcar": {imp: p.copy() for imp in IMPUTATIONS},
                               "tnr1|mcar": {imp: p.copy() for imp in IMPUTATIONS}, "tnr0|mcar": {imp: p.copy() for imp in IMPUTATIONS}}


def sort_dict(dictio):
    return collections.OrderedDict(sorted(dictio.items()))


"""{5: 'priors_count_is_Caucasian_synth|mar|spd|log_reg|cca',
 2: 'priors_count_is_Caucasian_synth|percentiles',
 3: 'priors_count_is_Caucasian_synth|log_reg_mcar_mice_def_0_0|Predicted true'}"""


def data_processing(data, models, percentiles):
    plotting_data = {k.split("|")[0]: {} for k in list(data.keys())}
    for key, value in data.items():
        div_key = key.split("|")
        if len(div_key) == 2:
            plotting_data["percentiles"] = [int(v) for v in value["mean"]]
        elif len(div_key) == 3:
            model_data = div_key[1].split("_")
            if model_data[0]+"_"+model_data[1] in models:
                model = model_data[0]+"_"+model_data[1]
                missing = model_data[2]
            else:
                model = model_data[0]
                missing = model_data[1]
            if model not in plotting_data[div_key[0]]:
                dict_initialiser(data=plotting_data, experiment=div_key[0], model=model,
                                 percentiles=percentiles)

            if div_key[-1] == "Predicted true":
                rate = "tpr"
                loc = 0
            else:
                rate = "tnr"
                loc = 1

            imp = model_data[-3] if model_data[-3] == "cca" or model_data[-3] == "mean" or model_data[-3] == "coldel"else str(
                model_data[-4])+"_"+str(model_data[-3])
            #print(imp)
            z = model_data[-1]
            missing_p = model_data[-2]
            #print(plotting_data[div_key[0]][model].keys())
            #print(model)
            plotting_data[div_key[0]][model][str(
                rate)+str(z)+"|"+missing][imp][missing_p] = value["mean"][loc]
        else:
            #print("HERE", plotting_data[div_key[0]][model][div_key[1]][div_key[-1]+"|"+div_key[2]])
            model = div_key[3]
            if model not in plotting_data[div_key[0]]:
                dict_initialiser(data=plotting_data, experiment=div_key[0], model=model,
                                 percentiles=percentiles)
            plotting_data[div_key[0]][model][div_key[1]
                                             ][div_key[-1]+"|"+div_key[2]] = value["mean"]
    return plotting_data


"""
metric_keys += [i+"|"+"acc" for i in IMPUTATIONS]
    data[experiment][model] = {"mar":{m:p.copy() for m in metric_keys}
{5: 'priors_count_is_Caucasian_synth|mar|spd|log_reg|cca',
 2: 'priors_count_is_Caucasian_synth|percentiles',
 3: 'priors_count_is_Caucasian_synth|log_reg_mcar_mice_def_0_0|Predicted true'}"""


def plotting_completer(title, xlabel, ylabel, save):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(Path(save))
    plt.clf()


CURRENT_MODELS = ["log_reg", "rf_cat", "knn"]


def plotting_multi(data, models, percentiles):
    full_table = {}
    plotting_data = data_processing(data, models, percentiles)
    #print(plotting_data['priors_count_is_Caucasian_synth'].keys())

    p = percentiles
    #print(p)
    for key, v in plotting_data.items():
        table = {m+imp: {}
                 for m, imp in zip(models, IMPUTATIONS) if m in CURRENT_MODELS}
        if key == "percentiles":
            continue
        if key is None:
            if not os.path.isdir(Path(SAVEPATH)):
                os.mkdir(Path(SAVEPATH))
            savepath = SAVEPATH
        if key is not None:
            if not os.path.isdir(Path(SAVEPATH)):
                os.mkdir(Path(SAVEPATH))
            if not os.path.isdir(Path(SAVEPATH+key+"/")):
                os.mkdir(Path(SAVEPATH+key+"/"))
            savepath = SAVEPATH+key+"/"
            print(savepath)

        for model, value in v.items():
            if model not in CURRENT_MODELS:
                continue
            for imp in IMPUTATIONS:
                table[model+imp] = {"Accuracy MAR 50%": -1, "Accuracy MAR 20%": -1,
                                    "Accuracy MCAR 50%": -1, "Accuracy MCAR 20%": -1,
                                    "SPD MAR 50%": -1, "SPD MAR 20%": -1, "SPD MCAR 50%": -1,
                                    "SPD MCAR 20%": -1, "PP MAR 50%": -1, "PP MAR 20%": -1,
                                    "PP MCAR 50%": -1, "PP MCAR 20%": -1}
            for miss in ["mar", "mcar"]:
                for x_axis in ["percent"]: # gives a bit weird plots and needs some work, "accuracy"]:
                    if x_axis == "percent":
                        x = p
                    else:
                        x = None
                    for imp in IMPUTATIONS:
                        if x is None:
                            x = value[miss][imp+"|"+"acc"]
                        spd = value[miss][imp+"|"+"spd"]
                        fig = plt.gcf()
                        fig.set_size_inches(18.5, 10.5)
                        plt.plot(x, spd, label="Imputation = "+NAME_KEYS[imp])

                        table[model+imp]["SPD "+miss.upper()+" 50%"] = spd[-1]
                        table[model+imp]["SPD "+miss.upper()+" 20%"] = spd[10]
                    plotting_completer(title="SPD for " + NAME_KEYS[model] +" with " + miss + " missingness", xlabel="Missingness percent",
                                       ylabel="SPD", save=savepath+model+"_"+miss+"_"+x_axis+"_spd.png")

                    for imp in IMPUTATIONS:
                        acc = value[miss][imp+"|"+"acc"]
                        #   print(acc)
                        fig = plt.gcf()
                        fig.set_size_inches(18.5, 10.5)
                        plt.plot(x, acc, label="Imputation = "+NAME_KEYS[imp])

                        table[model+imp]["Accuracy " +
                                         miss.upper()+" 50%"] = acc[-1]
                        table[model+imp]["Accuracy " +
                                         miss.upper()+" 20%"] = acc[10]

                    plotting_completer(title="Accuracy for " + NAME_KEYS[model]+" with " + miss + " missingness", xlabel="Missingness percent",
                                       ylabel="Accuracy", save=savepath+model+"_"+miss+"_"+x_axis+"_acc.png")

                    for imp in IMPUTATIONS:
                        if x is None:
                            x = value[miss][imp+"|"+"acc"]
                        pp = value["mar"][imp+"|"+"pp"]
                        fig = plt.gcf()
                        fig.set_size_inches(18.5, 10.5)
                        plt.plot(x, pp, label="Imputation = "+NAME_KEYS[imp])

                        table[model+imp]["PP "+miss.upper()+" 50%"] = pp[-1]
                        table[model+imp]["PP "+miss.upper()+" 20%"] = pp[10]

                    plotting_completer(title="Predictive Parity for " + NAME_KEYS[model]+" with " + miss + " missingness", xlabel="Missingness percent",
                                       ylabel="Predictive Parity", save=savepath+model+"_"+miss+"_"+x_axis+"_pp.png")
                #Adding to table

                for imp in IMPUTATIONS:
                    fig = plt.gcf()
                    fig.set_size_inches(18.5, 10.5)
                    #Plotting true positive and true negative rates against missingness
                    tpr1 = sort_dict(value["tpr1"+"|"+miss][imp])
                    tpr0 = sort_dict(value["tpr0"+"|"+miss][imp])
                    tnr1 = sort_dict(value["tnr1"+"|"+miss][imp])
                    tnr0 = sort_dict(value["tnr0"+"|"+miss][imp])
                    plt.plot(p,  list(tpr1.values()),
                             label="True Positive Rate with Z=1")
                    plt.plot(p,  list(tpr0.values()),
                             label="True Positive Rate with Z=0")
                    plt.plot(p,  list(tnr1.values()),
                             label="True Negative Rate with Z=1")
                    plt.plot(p,  list(tnr0.values()),
                             label="True Negative Rate with Z=0")

                    plt.title("True positive and negative rates for " +
                              NAME_KEYS[model] + " with imuptation= " + NAME_KEYS[imp])
                    plt.xlabel("Missingness percent")
                    plt.ylabel("True rate")
                    plt.ylim(0.4, 1.05)
                    plt.legend()
                    plt.savefig(Path(savepath+model+"_" +
                                miss+"_"+imp+"_rates.png"))
                    plt.clf()
        full_table[key] = table

    return full_table
