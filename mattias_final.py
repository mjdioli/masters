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

runs = 10



compas = {}
response_recid = "two_year_recid"
for rob in [True, False]:
    for miss in ["priors_count",'age_factor_Greater than 45', "crime_factor"]:#, "is_Caucasian", "gender_factor"]:
        for sensitive in ["is_Caucasian", "gender_factor"]:
            try:
                recid = utils_clean.run(response_recid, miss, sensitive, models = ["log_reg", "knn", "rf_cat"], dataset = "recid", n_runs = runs, robust=rob, with_mcar=True)
            except Exception as e:
                print("Failed on COMPAS at " + miss + " "+ sensitive + " with error: ", e)
            compas[miss+"|"+sensitive] = recid
            try:
                with open(Path(savepath+"compas.json"), 'w') as f:
                    json.dump(compas, f, indent=1)
            except:
                print("COULDNT SAVE COMPAS")
"""try:
    with open(Path(savepath+".json"), 'w') as f:
        json.dump(compas, f, indent=1)
except:
    print("COULDNT SAVE COMPAS")"""

synth_compas = {}
response_recid = "two_year_recid"
 
for miss in ['priors_count', 'age_factor_greater_than_45', "crime_factor"]:#, "is_Caucasian", "gender_factor"]:
    for sensitive in ["is_Caucasian", "gender_factor"]:
        try:
            synth_comp= utils_clean.run(response_recid, miss, sensitive, models = ["log_reg", "knn", "rf_cat"], dataset = "recid_synth", n_runs = runs, robust=True, with_mcar=True)
        except Exception as e:
            print("Failed on COMPAS SYNTH at " + miss + " "+ sensitive + " with error: ", e)
        synth_compas[miss+"|"+sensitive] = synth_comp
        try:
            with open(Path(savepath+"synth_compas.json"), 'w') as f:
                json.dump(synth_compas, f, indent=1)
        except:
            print("COULDNT SAVE SYNTH COMPAS")
        

        
    
synth_simple = {}
response_synth = "y"
sensitive = "x_1"
for miss in ["x_2", "x_5"]:
    try:
        simple = utils_clean.run(response_synth, miss, sensitive, models = ["log_reg", "knn", "rf_cat"], dataset = "simple", n_runs = runs, robust=True, with_mcar=True)
    except Exception as e:
        print("Failed on simple synth at " + miss + " "+ sensitive + " with error: ", e)
    synth_simple[miss+"|"+sensitive] = simple

    try:
        with open(Path(savepath+"simple.json"), 'w') as f:
            json.dump(synth_simple, f, indent=1)
    except:
        print("COULDNT SAVE SIMPLE SYNTH")
    
    
adult = {}
for miss in ["marital-status", "gender"]:
    for sensitive in ["gender", "race"]:
        if miss == sensitive:
            continue
        try:
            adult = utils_clean.run(response_recid, miss, sensitive, models = ["log_reg", "knn", "rf_cat"], dataset = "adult", n_runs = runs, robust=True, with_mcar=True)
        except Exception as e:
            print("Failed on adult at " + miss + " "+ sensitive + " with error: ", e)
        synth_simple[miss+"|"+sensitive] = adult

        try:
            with open(Path(savepath+"adult.json"), 'w') as f:
                json.dump(adult, f, indent=1)
        except:
            print("COULDNT SAVE ADULT")