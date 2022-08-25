import utils_clean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

savepath = "final_experiment/"
if not os.path.isdir(Path("final_experiment/")):
    os.mkdir(Path(savepath))
    
import json

alph = [1000,2.84]
imps = ["fair_reg_"+str(a) for a in range(15,100,15)]
runs = 10

print("COMPAS START")
compas = {}
response_recid = "two_year_recid"
for rob in [True, False]:
    for miss in ["priors_count"]:#, "is_Caucasian", "gender_factor"]:
        for sensitive in ["gender_factor"]:
            try:
                recid = utils_clean.run(response_recid, miss, sensitive, models = ["log_reg", "knn", "rf_cat"], dataset = "recid", n_runs = runs, robust=rob, with_mcar=True, imputation=imps, alphas=alph)
            except Exception as e:
                print("Failed on lambda_test at " + miss + " "+ sensitive + " with error: ", e)
            compas["robust_"+str(rob)+miss+"|"+sensitive] = recid
            try:
                with open(Path(savepath+"lambda_test.json"), 'w') as f:
                    json.dump(compas, f, indent=1)
            except:
                print("COULDNT SAVE lambda_test")
                