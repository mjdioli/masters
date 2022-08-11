import numpy as np
import utils
import pandas as pd
import os

synth_regular = utils.load_synthetic()
compas = utils.load_compas_alt()


percentiles = [0,5,10]+[p for p in range(20,80, 10)]
#percentiles = [p for p in range(60,80, 10)]

#Adult dataset
all_results = {"Full data": {}, "Averaged results": {} }
RUNS = 5
RESPONSE = "income"
for miss in ["workclass", "gender", "relationship"]:
    for sens in ["gender", "race"]:
        adult_results = utils.test_bench(data = "adult", pred = RESPONSE, missing = miss, sensitive=sens,
                        percentiles = percentiles, n_runs=RUNS, differencing=True)
        

        all_results["Full data"][miss+"_"+sens+"_"+"adult"] = adult_results["Full data"]
        all_results["Averaged results"][miss+"_"+sens+"_"+"adult"] = adult_results["Averaged results"]


#Simple synthetic
miss = "x_2"
sensitive = "x_1"
RESPONSE = "y"
synth_results = utils.test_bench(data = "simple", pred = RESPONSE, missing = miss, sensitive=sensitive,
                            percentiles = percentiles, n_runs=RUNS, differencing=True)
all_results["Full data"][miss+"_"+sensitive+"_"+"synth"] = synth_results["Full data"]
all_results["Averaged results"][miss+"_"+sensitive+"_"+"synth"] = synth_results["Averaged results"]

import json
from pathlib import Path
if not os.path.isdir(Path("raw_data/")):
    os.mkdir(Path("raw_data/"))
with open(Path("raw_data/intermediate.json"), 'w') as f:
    json.dump(all_results, f)
    
    
#Compas
RESPONSE = "two_year_recid"
for miss in ["crime_factor", "is_Caucasian", "gender_factor"]:
    for sensitive in ["is_Caucasian", "gender_factor"]:
        #try:
        recid_results = utils.test_bench(data = "compas", pred = RESPONSE, missing = miss, sensitive=sensitive,
                        percentiles = percentiles, n_runs=RUNS, differencing = True)
        synth_compas_results = utils.test_bench(data = "synthetic", pred = RESPONSE, missing = miss, sensitive=sensitive,
                            percentiles = percentiles, n_runs=RUNS, differencing = True)
        #TODO and remember to fix data v_recid_results = utils.test_bench(train = compas["standard"]["train"],test = compas["standard"]["test"], pred = RESPONSE, missing = miss, sensitive=sensitive,
                        #percentiles = percentiles)
        #all_results[miss+"_"+sensitive+"_"+"synth"] = synth_results 
        all_results["Full data"][miss+"_"+sensitive+"_"+"recid"] = recid_results["Full data"]
        all_results["Averaged results"][miss+"_"+sensitive+"_"+"recid"] = recid_results["Averaged results"]
        all_results["Full data"][miss+"_"+sensitive+"_"+"synth_compas"] = synth_compas_results["Full data"]
        all_results["Averaged results"][miss+"_"+sensitive+"_"+"synth_compas"] = synth_compas_results["Averaged results"]
        

with open(Path("raw_data/marius_data_2.json"), 'w') as f:
    json.dump(all_results, f)