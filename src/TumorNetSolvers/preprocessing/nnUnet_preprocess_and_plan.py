"""
Dataset Fingerprint extraction, experiment planning and preprocessing
Should occur after correct paths definition
Requires datasetName & int_id

"""
#%%
from TumorNetSolvers.utils.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results


print(nnUNet_preprocessed, nnUNet_raw, nnUNet_results)
id=600
from TumorNetSolvers.reg_nnUNet.utilities.dataset_name_id_conversion import find_candidate_datasets, maybe_convert_to_dataset_name
find_candidate_datasets(id)
dataset_name = maybe_convert_to_dataset_name(id)  

from TumorNetSolvers.reg_nnUNet.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
fpe = DatasetFingerprintExtractor(dataset_name, 8, verbose=True)
fingerprint=fpe.run()
#%%
#Experiment planning
from TumorNetSolvers.reg_nnUNet.experiment_planning.plan_and_preprocess_api import plan_experiment_dataset, preprocess_dataset
plan, plan_identifier=plan_experiment_dataset(id)
#%%
#Preprocess datatset 
from TumorNetSolvers.reg_nnUNet.experiment_planning.plan_and_preprocess_api import preprocess_dataset2
preprocess_dataset2(id,num_processes=(8, 4, 8))
