import utils_clean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

savepath = "big_run/"
if not os.path.isdir(Path("big_run/")):
    os.mkdir(Path(savepath))
    
import json

runs = 10


print("COMPAS START")
compas = {}
counter = 0
response_recid = "two_year_recid"
for rob in [True, False]:
    for miss in ["priors_count", "age_factor_Greater than 45", "crime_factor"]:#, "is_Caucasian", "gender_factor"]:
        for sensitive in ["is_Caucasian","gender_factor"]:
            if not rob:
                counter =0
            if counter > 3:
                continue
            if rob:
                counter = counter +1
            try:
                recid = utils_clean.run(response_recid, miss, sensitive, models = ["log_reg", "knn", "rf_cat"], dataset = "recid", n_runs = runs, robust=rob, with_mcar=True)
            except Exception as e:
                print("Failed on COMPAS at " + miss + " "+ sensitive + " with error: ", e)
            compas["robust_"+str(rob)+miss+"|"+sensitive] = recid
            try:
                with open(Path(savepath+"compas_10.json"), 'w') as f:
                    json.dump(compas, f, indent=1)
            except:
                print("COULDNT SAVE COMPAS")
"""try:
    with open(Path(savepath+".json"), 'w') as f:
        json.dump(compas, f, indent=1)
except:
    print("COULDNT SAVE COMPAS")"""


print("SYNTH COMPAS START")
synth_compas = {}
response_recid = "two_year_recid"
 
for miss in ['priors_count']:#, 'age_factor_greater_than_45', "crime_factor"]:#, 'priors_count', "is_Caucasian", "gender_factor"]:
    for sensitive in ["is_Caucasian"]:#, "gender_factor"]:
        try:
            synth_comp= utils_clean.run(response_recid, miss, sensitive, models = ["log_reg", "knn", "rf_cat"], dataset = "recid_synth", n_runs = runs, robust=False, with_mcar=True)
        except Exception as e:
            print("Failed on COMPAS SYNTH at " + miss + " "+ sensitive + " with error: ", e)
        synth_compas[miss+"|"+sensitive] = synth_comp
        try:
            with open(Path(savepath+"synth_compas_10.json"), 'w') as f:
                json.dump(synth_compas, f, indent=1)
        except:
            print("COULDNT SAVE SYNTH COMPAS")
        

        
print("SIMPLE SYNTHETIC START")
synth_simple = {}
response_synth = "y"
sensitive = "x_1"
for miss in ["x_2", "x_5"]:
    try:
        simple = utils_clean.run(response_synth, miss, sensitive, models = ["log_reg", "knn", "rf_cat"], dataset = "simple", n_runs = runs, robust=False, with_mcar=True)
    except Exception as e:
        print("Failed on simple synth at " + miss + " "+ sensitive + " with error: ", e)
    synth_simple[miss+"|"+sensitive] = simple

    try:
        with open(Path(savepath+"simple_10.json"), 'w') as f:
            json.dump(synth_simple, f, indent=1)
    except:
        print("COULDNT SAVE SIMPLE SYNTH")
    

print("ADULT START")
response_adult = "income"    
adult_full = {}
for miss in ["marital-status", "gender"]:
    for sensitive in ["gender", "race"]:
        if miss == sensitive:
            continue
        try:
            adult_data = utils_clean.run(response_adult, miss, sensitive, models = ["log_reg", "knn", "rf_cat"], dataset = "adult", n_runs = runs, robust=False, with_mcar=True)
        except Exception as e:
            print("Failed on adult at " + miss + " "+ sensitive + " with error: ", e)
        adult_full[miss+"|"+sensitive] = adult_data

        try:
            with open(Path(savepath+"adult_10.json"), 'w') as f:
                json.dump(adult_full, f, indent=1)
        except:
            print("COULDNT SAVE ADULT")