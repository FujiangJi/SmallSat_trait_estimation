import numpy as np
import pandas as pd
import os
import sys
from scipy import stats
import pickle
import datetime
import matplotlib.pyplot as plt
from osgeo import gdal,gdalconst
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
import warnings
warnings.filterwarnings('ignore')

def read_tif(tif_file):
    dataset = gdal.Open(tif_file)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    im_proj = (dataset.GetProjection())
    im_Geotrans = (dataset.GetGeoTransform())
    im_data = dataset.ReadAsArray(0, 0, cols, rows)
    if im_data.ndim == 3:
        im_data = np.moveaxis(dataset.ReadAsArray(0, 0, cols, rows), 0, -1)
    return im_data, im_Geotrans, im_proj,rows, cols

def array_to_geotiff(array, output_path, geo_transform, projection, band_names=None):
    rows, cols, num_bands = array.shape
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(output_path, cols, rows, num_bands, gdal.GDT_Float32)
    
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    
    for band_num in range(num_bands):
        band = dataset.GetRasterBand(band_num + 1)
        band.WriteArray(array[:, :, band_num])
        band.FlushCache()
        
        if band_names:
            band.SetDescription(band_names[band_num])
    dataset = None
    band = None
    return

#*********************************Start training*********************************#

target_wvl = np.arange(400,2401,10)
bad_bands = [[1320, 1440], [1770, 2040]]
exclude_indices = []
for band_range in bad_bands:
    indices = np.where((target_wvl >= band_range[0]) & (target_wvl <= band_range[1]))[0]
    exclude_indices.extend(indices)

exclude_indices = np.array(exclude_indices)
out_wvl = target_wvl
target_wvl = np.delete(target_wvl, exclude_indices)

out_path = "/mnt/cephfs/scratch/groups/chen_group/FujiangJi/SmallSat_data/4_random_modeling_results/"
HR_image_path = "/mnt/cephfs/scratch/groups/chen_group/FujiangJi/SmallSat_data/1_imagery_data/1_fused_imagery/"
LR_image = "/mnt/cephfs/scratch/groups/chen_group/FujiangJi/SmallSat_data/1_imagery_data/EMIT_L2A_RFL_20230422_clipped_resampled.tif"
HR_image = "/mnt/cephfs/scratch/groups/chen_group/FujiangJi/SmallSat_data/1_imagery_data/PlanetScope_RFL_20230422_clipped_resampled.tif"

models = ["EMIT", "Planet", "DSSFNET", "MSHFNET", "MSAHFNET", "SSFCNN", "TFNET", "CONSSFCNN", "RESTFNET", "MSDCNN", "SSRNET"]

traits = ['LWC_area', 'phosphorus', 'hemicellulose', 'cellulose', 'chl_area', 'phenolics_mg_g', 'LWC','potassium', 'sulfur', 'NSC_DS', 'nitrogen', 'lignin', 'LMA', '%C']

for arch in models:
    export_path = f"{out_path}/{arch}/"
    
    if (arch == "EMIT"):
        image = LR_image
    elif arch == "Planet":
        image = HR_image
    elif arch == "MSHFNET":
        image = f"{HR_image_path}Fusion3_model_fused_imagery.tif"
    elif arch == "MSAHFNET":
        image = f"{HR_image_path}Fusion5_model_fused_imagery.tif"
    else:
        image = f"{HR_image_path}{arch}_model_fused_imagery.tif"
        
    im_data, im_Geotrans, im_proj,_, _ = read_tif(image)
    im_array = im_data.reshape(-1, im_data.shape[2])
    
    if arch != "Planet":
        im_array = np.delete(im_array, exclude_indices, axis = 1)
    
    for tr in traits:
        model_path = f"{export_path}/saved_models/"
        n_iterations = 20
        start_var = True
        for iteration in range(n_iterations):
            with open(f"{model_path}/{tr}_PLSR_model_interation{iteration+1}.pkl", 'rb') as model:
                pls = pickle.load(model)
                
            pred = pls.predict(im_array)
            pred = pred.reshape(im_data.shape[0],im_data.shape[1], 1)
            if start_var:
                results = pred
                start_var = False
            else:
                results = np.concatenate((results, pred),axis = 2)

        mean_pred = results.mean(axis = 2)
        mean_std = results.std(axis = 2) 
        final = np.stack([mean_pred, mean_std], axis = 2)

        output_path = f"{export_path}{tr}_map.tif"
        array_to_geotiff(final, output_path, im_Geotrans, im_proj, band_names=[f"{tr}_mean", f"{tr}_std"]) 
