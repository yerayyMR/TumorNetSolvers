import os
import numpy as np
import torch
from torch.utils.data import  DataLoader
from TumorNetSolvers.inference.inference_manager import InferenceManager
from TumorNetSolvers.inference.inference_utils import CustomDataset, get_settings_and_file_paths

nnUNet_preprocessed = os.environ.get('nnUNet_preprocessed')
nnUNet_results = os.environ.get('nnUNet_results')

# If they are not set, raise an error or handle the missing paths
if not nnUNet_preprocessed or not nnUNet_results:
    raise EnvironmentError("One or more environment variables (nnUNet_preprocessed, nnUNet_results) are not set.")

def folder_is_empty_or_missing(folder_path: str) -> bool:
    return not os.path.exists(folder_path) or len(os.listdir(folder_path)) == 0


def run_inference(dataset_name: str, models: list, data_folder: str, output_base: str, device: torch.device = torch.device('cuda:0'), signature: str = '10k', chkpt: str=''):
    """
    Runs inference on a given dataset using one or more models & saves predictions to specified output 
    directory. Optionally, saves masked outputs (useful for performance comparison with baseline pipeline).

    Args:
        dataset_name (str): e.g., "Dataset700_Brain"
        models (list): List of model names to use for inference (choices: ['ViT', 'nnUnet', 'TumorSurrogate']).
        data_folder (str): Path to the folder containing the input data.
        output_base (str): Path to the folder where results will be saved.
        device (torch.device, optional): device to run the inference on 
        signature (str, optional): Unique identifier for model signature (default is '10k').
        chkpt (str, optional): gives the possibility of inputing an arbitrary checkpoint path (and not necessarily model/best_ema_checkpoint)
    Returns:
        None: Saves inference results to the specified output directory.
    """
    # Validate models list
    valid_models = ['ViT', 'nnUnet', 'TumorSurrogate']
    for model in models:
        if model not in valid_models:
            raise ValueError(f"Invalid model '{model}'. Valid models are: {valid_models}")

    # Load settings and file paths
    plan, dataset_json, test_keys, parameters = get_settings_and_file_paths(dataset_name)

    # Set up the dataset and data loader
    dataset = CustomDataset(data_folder, test_keys)
    data_loader = DataLoader(dataset, batch_size=5, shuffle=False)
    os.makedirs(output_base, exist_ok=True)
    # Create output directories for each model
    for model_name in models:
        os.makedirs(os.path.join(output_base, f"_{model_name}_{signature}/masked"), exist_ok=True)
        os.makedirs(os.path.join(output_base, f"_{model_name}_{signature}/notMasked"), exist_ok=True)
        masked_gt_folder = os.path.join(nnUNet_results,dataset_name, f'masked_gt')
        save_masked_gt = folder_is_empty_or_missing(masked_gt_folder)

        if save_masked_gt:
            os.makedirs(masked_gt_folder, exist_ok=True)
    # Perform inference for each model
    for model_name in models:
        # Initialize trainer and load model checkpoint
        infer_manager = InferenceManager(plan, configuration='3d_fullres', model=model_name, device=device, dataset_json=dataset_json)#, signature=signature)

        # Construct the checkpoint path
        if chkpt and os.path.exists(chkpt):
            checkpoint_path=chkpt
        else:
            checkpoint_path = os.path.join(nnUNet_results, dataset_name, 'Trainer__nnUNetPlans__3d_fullres', 'fold_train_val',
                                       f'_{signature}_{model_name}', f'checkpoint_{model_name}_best_ema_loss.pth')
        infer_manager.load_checkpoint(checkpoint_path)
        model = infer_manager.network._orig_mod.to(device).eval()

        # Inference loop
        with torch.no_grad():
            for batch in data_loader:
                # Check the length of the batch
                if len(batch) == 4:
                    data, mask, target, keys = batch 
                elif len(batch) == 3:
                    data, _, keys = batch  
                    mask=None
                  
                else:
                    raise ValueError("Unexpected batch structure")

                data = data.to(device, non_blocking=True)
                if mask is not None:
                    mask = mask.to(device)

                # Gather model parameters and perform inference
                batch_params = [parameters[key] for key in keys]
                batch_params = torch.stack(batch_params).to(device)
                output = model(data, batch_params)

                # Apply deep supervision if enabled
                if isinstance(output,list):
                    output = output[0]

                # Save predictions
                for i in range(output.shape[0]):
                    case_n = keys[i]
                    np.save(os.path.join(output_base, f"_{model_name}_{signature}/notMasked", f'{case_n}.npy'), output[i].cpu().numpy())
                    if mask is not None:
                        np.save(os.path.join(output_base, f"_{model_name}_{signature}/masked", f'{case_n}.npy'), (output[i].cpu().numpy() * mask[i].cpu().numpy()))
                        if save_masked_gt:
                            #nevermind the word "seg", this is still a regression task. {case_n}_seg.npy files are gt simulations, we're just using this 
                            # naming format to maintain compatibility with the code from batchgenerators module and avoid further unnecessary adjustments
                            np.save(os.path.join(masked_gt_folder, f'{case_n}_seg.npy'), target[i].cpu().numpy() * mask[i].cpu().numpy())

