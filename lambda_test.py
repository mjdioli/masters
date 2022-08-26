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

#REMEMBER THAT I HAVE CHANGED NUMBER OF ITERATIONS IN UTILS CLEAN FOR THE FAIR LOG REG.

alph = [1000,2.84]
imps = ["fair_reg_"+str(a) for a in range(20,100,20)] + ["fair_reg_5","fair_reg_50","fair_reg_95"]
runs = 10

print("COMPAS START")
compas = {}
response_recid = "two_year_recid"
for rob in [True, False]:
    for miss in ["priors_count"]:#, "is_Caucasian", "gender_factor"]:
        for sensitive in ["gender_factor"]:
            try:
                recid = utils_clean.run(response_recid, miss, sensitive, models = ["log_reg", "knn", "rf_cat"], dataset = "recid", n_runs = runs, robust=rob, with_mcar=True, imputation=imps, alphas=alph, missing_percentages =[0,50] )
            except Exception as e:
                print("Failed on lambda_test at " + miss + " "+ sensitive + " with error: ", e)
            compas[str(rob)+"|"+miss+"|"+sensitive] = recid
            try:
                with open(Path(savepath+"compas_lambda_test2.json"), 'w') as f:
                    json.dump(compas, f, indent=1)
            except:
                print("COULDNT SAVE lambda_test")

print("SYNTH COMPAS START")
synth_compas = {}
response_recid = "two_year_recid"
for rob in [True, False]:
    for miss in ['priors_count']:#, 'age_factor_greater_than_45', "crime_factor"]:#, 'priors_count', "is_Caucasian", "gender_factor"]:
        for sensitive in ["is_Caucasian"]:#, "gender_factor"]:
            try:
                synth_comp= utils_clean.run(response_recid, miss, sensitive, models = ["log_reg", "knn", "rf_cat"], dataset = "recid_synth", n_runs = runs, robust=rob, with_mcar=True)
            except Exception as e:
                print("Failed on COMPAS SYNTH at " + miss + " "+ sensitive + " with error: ", e)
            synth_compas[str(rob)+"|"+miss+"|"+sensitive] = synth_comp
            try:
                with open(Path(savepath+"synth_compas_lambda_test.json"), 'w') as f:
                    json.dump(synth_compas, f, indent=1)
            except:
                print("COULDNT SAVE SYNTH COMPAS")