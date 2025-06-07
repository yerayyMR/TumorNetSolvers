import os
import json
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import concurrent.futures
import scipy.ndimage

def load_params(params_file, is_synthetic=True):
    """
    Load tumor simulation parameters from a JSON file.

    Args:
        params_file (str): Path to the parameters JSON file.
        is_synthetic (bool): Flag indicating whether the data is synthetic.

    Returns:
        tuple: (params dict, diffusion coefficient (Dw), growth rate (rho), final time (T)).
    """
    if is_synthetic:
        with open(params_file, 'r') as f:
            jsonFile = json.load(f)
        params = jsonFile['parameters']
        D = float(params['Dw'])
        rho = float(params['rho'])
        T = float(jsonFile['results']['final_time'])
        return params, D, rho, T
    else:
        # For real patient data, return zeros for parameters
        return {}, 0.0, 0.0, 0.0

def load_real_patient_data(patient_folder):
    """
    Load the tumor data for real patient from core and edema images.

    Args:
        patient_folder (str): Path to the patient folder containing core and edema images.

    Returns:
        ndarray: The generated tumor image as a weighted sum of core and edema.
    """
    core_img = nib.load(f"{patient_folder}/tumorCore_flippedCorrectly.nii").get_fdata()>0
    edema_img = nib.load(f"{patient_folder}/tumorFlair_flippedCorrectly.nii").get_fdata()>0

    # Combine the images to generate the tumor image (for example, a weighted sum)
    tumorImg =  (0.675 - 0.25) * core_img + 0.25 * edema_img  # Adjust the weights as needed
    return tumorImg
def compute_derived_params(D, rho, T):
    """
    Compute derived parameters for tumor growth dynamics.

    Args:
        D (float): Diffusion coefficient.
        rho (float): Growth rate.
        T (float): Simulation time.

    Returns:
        tuple: Square root of scaled diffusion (muD) and growth rate (muRho).
    """
    muD = np.sqrt(D * T).astype(np.float32)
    muRho = np.sqrt(rho * T).astype(np.float32)
    return muD, muRho


def compute_center_and_displacement(tumorImg, params):
    """
    Compute the center of mass and displacement of the tumor.

    Args:
        tumorImg (ndarray): Tumor simulation image.
        params (dict): Tumor growth parameters.

    Returns:
        tuple: (Image center, center of mass, displacement values in x, y, z).
    """
    NxT1_pct, NyT1_pct, NzT1_pct = params['NxT1_pct'], params['NyT1_pct'], params['NzT1_pct']
    [icx, icy, icz] = [tumorImg.shape[1] * NxT1_pct, tumorImg.shape[0] * NyT1_pct, tumorImg.shape[-1] * NzT1_pct]
    com = scipy.ndimage.center_of_mass(tumorImg)
    com = [int(i) for i in com]
    x = (icx - com[0]) / tumorImg.shape[0]
    y = (icy - com[1]) / tumorImg.shape[1]
    z = (icz - com[2]) / tumorImg.shape[-1]
    img_center = [tumorImg.shape[0] // 2, tumorImg.shape[1] // 2, tumorImg.shape[2] // 2]
    return img_center, com, x, y, z


def shift_images(atlasImg, tumorImg, img_center, com):
    """
    Shift images to align the tumor center with the image center.

    Args:
        atlasImg (ndarray): Tissue atlas image.
        tumorImg (ndarray): Tumor simulation image.
        img_center (list): Center of the image.
        com (list): Center of mass of the tumor.

    Returns:
        tuple: (Shifted atlas image, shifted tumor image).
    """
    rollX = img_center[0] - com[0]
    rollY = img_center[1] - com[1]
    rollZ = img_center[2] - com[2]
    shifted_atlasImg = torch.tensor(atlasImg).roll(shifts=(rollX, rollY, rollZ), dims=(0, 1, 2))
    shifted_tumorImg = torch.tensor(tumorImg).roll(shifts=(rollX, rollY, rollZ), dims=(0, 1, 2))
    return shifted_atlasImg, shifted_tumorImg


def crop_and_downsample(shifted_img, img_center, crop_size, downsample_size):
    """
    Crop and optionally downsample the image.

    Args:
        shifted_img (Tensor): Shifted image tensor.
        img_center (list): Center of the image.
        crop_size (int): Desired crop size (e.g., 120x120x120).
        downsample_size (int): Target downsample size (e.g., 64x64x64).

    Returns:
        Tensor: Cropped and downsampled image.
    """
    cropped_img = shifted_img[
        img_center[0] - crop_size // 2:img_center[0] + crop_size // 2,
        img_center[1] - crop_size // 2:img_center[1] + crop_size // 2,
        img_center[2] - crop_size // 2:img_center[2] + crop_size // 2
    ]

    if downsample_size is not None:
        output_size = (downsample_size, downsample_size, downsample_size)
        downsampled_img = F.interpolate(cropped_img.unsqueeze(0).unsqueeze(1), size=output_size, mode='trilinear')
        return downsampled_img.squeeze(0).squeeze(0)

    return cropped_img



def data_preprocess(tumorImg, atlasImg, params_file, crop_sz, downsample_sz, is_synthetic=True):
    """
    Preprocess tumor and atlas images by computing displacements, 
    shifting, cropping, and downsampling.

    Args:
        tumorImg (ndarray): Tumor simulation image.
        atlasImg (ndarray): Tissue atlas image.
        params_file (str): Path to the JSON file containing simulation parameters.
        crop_sz (int): Size for cropping the images.
        downsample_sz (int): Size for downsampling the images.
        is_synthetic (bool): Flag indicating whether the data is synthetic.

    Returns:
        tuple: 
            - Downsampled tumor image (Tensor).
            - Downsampled atlas image (Tensor).
            - List of derived parameters for the simulation.
    """
    params, D, rho, T = load_params(params_file, is_synthetic)
    muD, muRho = compute_derived_params(D, rho, T)
    img_center, com, x, y, z = compute_center_and_displacement(tumorImg, params)
    shifted_atlasImg, shifted_tumorImg = shift_images(atlasImg, tumorImg, img_center, com)
    downsampled_tumor = crop_and_downsample(shifted_tumorImg, img_center, crop_size=crop_sz, downsample_size=downsample_sz)
    downsampled_atlas = crop_and_downsample(shifted_atlasImg, img_center, crop_size=crop_sz, downsample_size=downsample_sz)
    param_list = [x, y, z, muD, muRho]
    return downsampled_tumor, downsampled_atlas, param_list


def process_patient(patient, atlasTissue, id, datasetPath, mount_dir, crop_sz, downsample_sz, anatomical_struct="Brain", is_synthetic=True):
    """
    Process an individual patient by loading data, preprocessing images, 
    and saving results in the required format.

    Args:
        patient (str): Name of the patient folder.
        atlasTissue (Nifti1Image): Tissue atlas image.
        id (int): Dataset ID.
        datasetPath (str): Path to the dataset directory.
        mount_dir (str): Mount directory for output.
        crop_sz (int): Size for cropping images.
        downsample_sz (int): Size for downsampling images.
        anatomical_struct (str): Anatomical structure name, e.g., "Brain".
        is_synthetic (bool): Flag indicating if the data is synthetic or real.

    Returns:
        dict: Parameter dictionary for the patient.
    """
    try:
        patient_number = patient.split('_')[1].split('.')[0]
        patient_number_int = int(patient_number)
        formatted_patient_number = str(patient_number_int + 1)
        file_n1 = f"{anatomical_struct.upper()}_p{formatted_patient_number}_0000.nii.gz"
        file_n2 = f"{anatomical_struct.upper()}_p{formatted_patient_number}.nii.gz"
        file_n3 = f"{anatomical_struct.upper()}_p{formatted_patient_number}"
        print(f"Processing patient {patient_number}, saving as {file_n3}")

        patientImg = nib.load(f"{datasetPath}{patient}/tumor_concentration.nii.gz")
        tumorImg = patientImg.get_fdata() if is_synthetic else load_real_patient_data(f"{datasetPath}{patient}")   ####
        atlasImg = atlasTissue.get_fdata()

        file = f"{datasetPath}{patient}/saveDict.json"
        sim, downsampled_atlas, param_list = data_preprocess(tumorImg, atlasImg, file, crop_sz, downsample_sz)

        param_dict = {file_n3: torch.tensor(param_list)}  

        inpt = nib.Nifti1Image(downsampled_atlas.squeeze(0).squeeze(0), atlasTissue.affine, atlasTissue.header)
        out = nib.Nifti1Image(sim.squeeze(0).squeeze(0), atlasTissue.affine, atlasTissue.header)

        os.makedirs(os.path.dirname(f"{mount_dir}/raw_data/Dataset{id}_{anatomical_struct}/imagesTr/{file_n1}"), exist_ok=True)
        os.makedirs(os.path.dirname(f"{mount_dir}/raw_data/Dataset{id}_{anatomical_struct}/labelsTr/{file_n2}"), exist_ok=True)

        nib.save(inpt, f"{mount_dir}/raw_data/Dataset{id}_{anatomical_struct}/imagesTr/{file_n1}")
        nib.save(out, f"{mount_dir}/raw_data/Dataset{id}_{anatomical_struct}/labelsTr/{file_n2}")

        return param_dict  
    except Exception as e:
        print(f"An error occurred while processing patient {patient}: {e}")
        return {}


def preparingDataset(id, mount_dir, anatomical_struct="Brain", start=0, stop=9, crop_sz=120, downsample_sz=64, is_synthetic=True):
    """
    Prepare the dataset by processing multiple patients in parallel.

    Args:
        id (int): Dataset ID for output directories.
        mount_dir (str): Mount directory for storing outputs.
        anatomical_struct (str): Anatomical structure name, (default: "Brain").
        start (int): Starting index for patient processing.
        stop (int): Ending index for patient processing.
        crop_sz (int): Size for cropping images.
        downsample_sz (int): Size for downsampling images.
        is_synthetic (bool): Flag indicating synthetic or real data.
        

    Returns:
        dict: Combined parameter dictionary for all patients.
    """
        
    datasetPath = "/mnt/Drive4/jonas/datasets/synthetic_FK_Michals_solver_smaller/"
    atlasTissue = nib.load("/home/home/yeray_jonas/tumornetsolvers/sub-mni152_tissues_space-sri.nii.gz")
    patients = np.sort(os.listdir(datasetPath))

    if start < 0:
        start = 0
    if stop > len(patients):
        stop = len(patients)

    patients_to_process = patients[start:stop]
    all_param_dicts = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_patient, patient, atlasTissue, id, datasetPath, mount_dir, crop_sz, downsample_sz, anatomical_struct, is_synthetic)
            for patient in patients_to_process
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                param_dict = future.result()
                all_param_dicts.append(param_dict)
            except Exception as e:
                print(f"An error occurred: {e}")

    combined_param_dict = {k: v for d in all_param_dicts for k, v in d.items()} 
    return combined_param_dict  


def create_json_file(param_dict, raw_dataset_path, comment=""):
    """
    Create a dataset JSON file summarizing the processed data.

    Args:
        param_dict (dict): Combined parameter dictionary for all patients.
        raw_dataset_path (str): Path to the raw dataset directory.
        comment (str): Optional comment to include in the JSON file.
    """
    data = {
        "channel_names": {
            "0": "MRI"
        },
        "numTraining": len(param_dict),
        "file_ending": ".nii.gz",
        "_comment": comment
    }
    f_name = os.path.join(raw_dataset_path, "dataset.json")
    with open(f_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"JSON file '{f_name}' created successfully.")
