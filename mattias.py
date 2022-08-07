import numpy as np
import utils
import pandas as pd

synth_regular = utils.load_synthetic()
compas = utils.load_compas_alt()
RESPONSE = "two_year_recid"

percentiles = [p for p in range(0,20,2)]+[p for p in range(20,80, 10)]
#TODO test missing = Sensitive in case code crashes
missing=["two_year_recid", "crime_factor", "is_Caucasian"]#, "gender_factor", 
all_results = {"Full data": {}, "Averaged results": {} }
RUNS = 10
for miss in missing:
    for sensitive in ["is_Caucasian", "gender_factor"]:
        #try:
        recid_results = utils.test_bench(data = "compas", pred = RESPONSE, missing = miss, sensitive=sensitive,
                        percentiles = percentiles, n_runs=RUNS)
        synth_results = utils.test_bench(data = "synthetic", pred = RESPONSE, missing = miss, sensitive=sensitive,
                            percentiles = percentiles, n_runs=RUNS)
        #TODO and remember to fix data v_recid_results = utils.test_bench(train = compas["standard"]["train"],test = compas["standard"]["test"], pred = RESPONSE, missing = miss, sensitive=sensitive,
                        #percentiles = percentiles)
        #all_results[miss+"_"+sensitive+"_"+"synth"] = synth_results 
        all_results["Full data"][miss+"_"+sensitive+"_"+"recid"] = recid_results["Full data"]
        all_results["Averaged results"][miss+"_"+sensitive+"_"+"recid"] = recid_results["Averaged results"]
        
import json
from pathlib import Path
with open(Path("raw_data/test.json"), 'w') as f:
    json.dump(all_results, f)