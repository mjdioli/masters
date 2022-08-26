import utils_clean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

savepath = "50_run_average/"
if not os.path.isdir(Path("50_run_average/")):
    os.mkdir(Path(savepath))
    
import json

runs = 50
recid_alpha= [1.033]
simple_alpha = [0.1]
adult_alpha = [4.92]
missing_pct = [90]

"""print("COMPAS START")
compas = {}
response_recid = "two_year_recid"
for rob in [True, False]:
    for miss in ["crime_factor"]:#, "is_Caucasian", "gender_factor"]:
        for sensitive in ["is_Caucasian","gender_factor"]:
        
            recid = utils_clean.run(response_recid, miss, sensitive, models = ["log_reg", "knn", "rf_cat"],
                                        dataset = "recid", n_runs = runs, robust=rob, with_mcar=True, missing_percentages=missing_pct, alphas=recid_alpha)
            #except Exception as e:
                #print("Failed on COMPAS at " + miss + " "+ sensitive + " with error: ", e)
            compas[str(rob)+"|"+miss+"|"+sensitive] = recid
            try:
                with open(Path(savepath+"compas_50_run_90pct.json"), 'w') as f:
                    json.dump(compas, f, indent=1)
            except:
                print("COULDNT SAVE COMPAS")"""
"""try:
    with open(Path(savepath+".json"), 'w') as f:
        json.dump(compas, f, indent=1)
except:
    print("COULDNT SAVE COMPAS")"""


print("SYNTH COMPAS START")
synth_compas = {}
response_recid = "two_year_recid"
for rob in [True, False]:
    for miss in ["crime_factor"]:#, 'priors_count', "is_Caucasian", "gender_factor"]:
        for sensitive in ["is_Caucasian", "gender_factor"]:
            try:
                synth_comp= utils_clean.run(response_recid, miss, sensitive, models = ["log_reg", "knn", "rf_cat"], dataset = "recid_synth",
                                            n_runs = runs, robust=False, with_mcar=True, missing_percentages=missing_pct, alphas=recid_alpha)
            except Exception as e:
                synth_comp = {}
                print("Failed on COMPAS SYNTH at " + miss + " "+ sensitive + " with error: ", e)
            synth_compas[str(rob)+"|"+miss+"|"+sensitive] = synth_comp
            try:
                with open(Path(savepath+"synth_compas_cf_50_run_90pct.json"), 'w') as f:
                    json.dump(synth_compas, f, indent=1)
            except:
                print("COULDNT SAVE SYNTH COMPAS")
        

        
print("SIMPLE SYNTHETIC START")
synth_simple = {}
response_synth = "y"
sensitive = "x_1"
for rob in [True, False]:
    for miss in ["x_2"]:
        try:
            simple = utils_clean.run(response_synth, miss, sensitive, models = ["log_reg", "knn", "rf_cat"],
                                     dataset = "simple", n_runs = runs, robust=False, with_mcar=True, missing_percentages=missing_pct, alphas=simple_alpha)
        except Exception as e:
            simple = {}
            print("Failed on simple synth at " + miss + " "+ sensitive + " with error: ", e)
        synth_simple[str(rob)+"|"+miss+"|"+sensitive] = simple

        try:
            with open(Path(savepath+"simple_50_run_90pct.json"), 'w') as f:
                json.dump(synth_simple, f, indent=1)
        except:
            print("COULDNT SAVE SIMPLE SYNTH")
    

print("ADULT START")
response_adult = "income"    
adult_full = {}
for rob in [True, False]:
    for miss in ["marital-status", "gender"]:
        for sensitive in ["race"]:
            if miss == sensitive:
                continue
            try:
                adult_data = utils_clean.run(response_adult, miss, sensitive, models = ["log_reg", "knn", "rf_cat"],
                                             dataset = "adult", n_runs = runs, robust=False, with_mcar=True, missing_percentages=missing_pct, alphas=adult_alpha)
            except Exception as e:
                adult_data = {}
                print("Failed on adult at " + miss + " "+ sensitive + " with error: ", e)
            adult_full[miss+"|"+sensitive] = adult_data

            try:
                with open(Path(savepath+"adult_ms_50_run_90pct.json"), 'w') as f:
                    json.dump(adult_full, f, indent=1)
            except:
                print("COULDNT SAVE ADULT")