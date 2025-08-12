'''
This file MUST be run through an interactive window (Mayus + Enter on sections).
Currently plots are not being saved elsewhere.
'''



# %%
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
src_path = os.path.abspath(os.path.join(current_dir, '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from TumorNetSolvers.post_analysis.plotting import plt_coeffs, plt_dice_scores, plt_diff, plt_diff_individual, plt_rho_D_comp, plt_error_volume_rho_D
from TumorNetSolvers.post_analysis.additional_plotting import mul_histo_comp, worst_best_error
from TumorNetSolvers.post_analysis.utils import calc_audc, num_params, save_full_ground_truths
from set_env import set_environment_variables
set_environment_variables() # Modify path in this function accordingly
nnUNet_preprocessed = os.getenv('nnUNet_preprocessed')
nnUNet_results = os.getenv('nnUNet_results')
#nnUNet_results = os.path.join('/mnt/Drive4/yeray_jonas/TumorNetSolvers_ext/data_and_outputs/', "results")
#
def main():

    DATASET_NAME = 'Dataset900_Brain'       # Specify the dataset name
    ENDING = 'trial'                           # Specific ending to the naming of the folder where weights and logs will be saved (str or None) -- If None default naming based one experiment will be used
    SIGNATURE='10k'

    # Models and experiments for overall plots. Here the options are presented, each will need to be defined for each plot based on interest.

    '''MODEL_EXPERIMENTS = {

        'nnUnet': [
            ['c', 'a_downsampling'], ['a', 'a_downsampling'],
            ['c', 'b_downsampling'], ['a', 'b_downsampling'],
            ['c', 'inputs'], ['a', 'inputs'],
            ['c', 'b_bottleneck'], ['a', 'b_bottleneck'],
            ['c', 'a_bottleneck'], ['a', 'a_bottleneck'],
            ['c', 'a_upsampling'], ['a', 'a_upsampling'],
            ['c', 'b_upsampling'], ['a', 'b_upsampling'],
            ['a', 'a_upsampling_skip']
        ],
        'TumorSurrogate': [
            ['c', 'a_downsampling'], ['a', 'a_downsampling'],
            ['c', 'b_downsampling'], ['a', 'b_downsampling'],
            ['c', 'inputs'], ['a', 'inputs'],
            ['c', 'b_bottleneck'], ['a', 'b_bottleneck'],
            ['c', 'a_bottleneck'], ['a', 'a_bottleneck'],
            ['c', 'a_bottleneck_after'], ['a', 'a_bottleneck_after'],
            ['c', 'a_upsampling_after'], ['a', 'a_upsampling_after'],
        ],
        'ViT': [
            ['MLP', 'one_token'],
            ['Linear', 'one_token'],
            ['MLP', 'mul_token'],
            ['Linear', 'mul_token'],
            ['MLP', 'embed_concat'],
            ['Linear', 'embed_concat'],
            ['MLP', 'embed_add'],
            ['Linear', 'embed_add']
        ]
    }'''
    
    
    # On the following, each plot will be defined with a True/False variable to consider it or not and in each section,
    # the specific extra required values will be asked per section if needed

    #########
    COEFF_PLOT = False
    # The following plot will create a comparison for the different coefficients (same coefficient applied to all PDE coefficients)
    # Furthremore it can be defined for each patient what slicing plane is desired. -- Automatically done on the center of mass of the tumor

    if COEFF_PLOT:

        MODEL_EXPERIMENTS = {

            'nnUnet': [
                ['c', 'a_upsampling']
            ],
            'TumorSurrogate': [
                ['c', 'a_bottleneck_after']
            ],
            'ViT': [
                ['Linear', 'embed_concat']
            ]
        }

        PATIENT_IDS = ["BRAIN_p875", "BRAIN_p9071", "BRAIN_p3650"]  # Add more patient IDs as needed

        # Define slicing plane per patient
        # Options: "axial", "coronal", "sagittal"
        PLANE_PER_PATIENT = {
            "BRAIN_p875": "axial",
            "BRAIN_p9071": "coronal",
            "BRAIN_p3650": "sagittal"
        }

        COEFFICIENTS = [[0], [0.2], [0.4], [0.6], [0.8], [1]]   # (ONLY on this format) Which percentages shall be considered (refer to "running_infernece.py" for background knowledge)

        plt_coeffs(nnUNet_results, PATIENT_IDS, PLANE_PER_PATIENT, MODEL_EXPERIMENTS, DATASET_NAME, SIGNATURE, COEFFICIENTS, ENDING)


    #########
    DIFF_PLOT = False
    DIFF_PLOT_INDIVIDUAL = False
    # Both plots create the difference among the prediction and ground truth.
    # "INDIVIDUAL" is in case of interest the plots are provided one for each architecture instead of a common one.

    if DIFF_PLOT or DIFF_PLOT_INDIVIDUAL:
        # Models and experiments need to be specifically defined since 
        MODELS = ["nnUnet", "TumorSurrogate", "ViT"]
        EXPERIMENTS = [["c", "a_upsampling"], ['c', 'a_bottleneck_after'], ['Linear', 'embed_concat']]
        ARCH_LABELS = ["U-Net", "TS", "ViT"]

        if DIFF_PLOT:

            PATIENT_ID = "BRAIN_p875" #1973#1911#12726

            plt_diff(nnUNet_results, MODELS, EXPERIMENTS, PATIENT_ID, ARCH_LABELS, DATASET_NAME, SIGNATURE, ENDING)
        elif DIFF_PLOT_INDIVIDUAL:

            PATIENT_IDS = {
                "nnUnet": "BRAIN_p3650",
                "TumorSurrogate": "BRAIN_p3650",
                "ViT": "BRAIN_p3650"
            }

            plt_diff_individual(nnUNet_results, MODELS, EXPERIMENTS, PATIENT_IDS, ARCH_LABELS, DATASET_NAME, SIGNATURE, ENDING)

    
    #########
    DICE_PLOTS = False
    # Generates a Dice Score plot for each of the networks and one combined

    if DICE_PLOTS:

        MODEL_EXPERIMENTS = {

            'nnUnet': [
                ['c', 'a_upsampling']
            ],
            'TumorSurrogate': [
                ['c', 'a_bottleneck_after']
            ],
            'ViT': [
                ['Linear', 'embed_concat']
            ]
        }

        plt_dice_scores(MODEL_EXPERIMENTS, DATASET_NAME, ENDING, SIGNATURE)

    #########
    ERROR_VOL_RHO_D = False
    # The following plot creates on the same plot for each of the architectures and corresponding experiment
    # the error over volume and the rho vs D plots.

    if ERROR_VOL_RHO_D:

        MODEL_EXPERIMENTS = {
            #'nnUnet': [['c', 'a_upsampling']],
            'TumorSurrogate': [['c', 'a_bottleneck_after']],
            'ViT': [['Linear', 'embed_concat']]
        }

        plt_error_volume_rho_D(DATASET_NAME, MODEL_EXPERIMENTS, ENDING, SIGNATURE)


    #########
    RHO_VS_D_COMP = True
    # The following will plot a comparison on different combinations of percentages of the original coefficients
    if RHO_VS_D_COMP:

        PATIENT_ID = "p875"
        DATASET_NAME = "Dataset900_Brain"

        MODEL_EXPERIMENTS = {
            'nnUnet': [['c', 'a_upsampling']],
            #'TumorSurrogate': [['c', 'a_bottleneck_after']],
            #'ViT': [['Linear', 'embed_concat']]
        }

        # Coefficient grid
        COEFFICIENTS = [
            [0, 0], [0, 0.25], [0, 0.5], [0, 0.75], [0, 1],
            [0.25, 0], [0.25, 0.25], [0.25, 0.5], [0.25, 0.75], [0.25, 1],
            [0.5, 0], [0.5, 0.25], [0.5, 0.5], [0.5, 0.75], [0.5, 1],
            [0.75, 0], [0.75, 0.25], [0.75, 0.5], [0.75, 0.75], [0.75, 1],
            [1, 0], [1, 0.25], [1, 0.5], [1, 0.75], [1, 1]
        ]

        plt_rho_D_comp(nnUNet_results, COEFFICIENTS, DATASET_NAME, MODEL_EXPERIMENTS, PATIENT_ID, ENDING, SIGNATURE)

if __name__ == "__main__":
    main()
# %%
