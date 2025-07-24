import numpy as np
import pandas as pd
import os
import sys
from scipy import stats
import pickle
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
import warnings
warnings.filterwarnings('ignore')

def rsquared(x, y): 
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y) 
    a = r_value**2
    return a
    
def vip(x, y, model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    
    m, p = x.shape
    _, h = t.shape
    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)
    return vips

def press(train_X,train_y,test_X,test_y, tr):
    press_scores = []
    for i in np.arange(1,train_X.shape[1]+1):
    # for i in np.arange(1,8):
        pls = PLSRegression(n_components=i)
        pls.fit(train_X, train_y)
        
        pred = pls.predict(test_X)
        aa = np.array(pred.reshape(-1,).tolist())
        bb = np.array(test_y.tolist())
        score = np.sum((aa - bb) ** 2)
        press_scores.append(score)
    n_components = press_scores.index(min(press_scores))+1
    print(f"   - {tr} model: random CV_n_components: {n_components}")
    press_scores = pd.DataFrame({'ncomp': np.arange(1,train_X.shape[1]+1), f'PRESS_score': press_scores})
    # press_scores = pd.DataFrame({'ncomp': np.arange(1,8), f'PRESS_score': press_scores})
    return press_scores, n_components

def random_CV(X, y, tr, n_iterations, target_wvl, out_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    press_scores, n_components = press(X_train, y_train, X_test, y_test, tr)
    
    plsr_coef = pd.DataFrame(np.zeros(shape = (n_iterations, X.shape[1])),columns = target_wvl)
    vip_score = pd.DataFrame(np.zeros(shape = (n_iterations, X.shape[1])),columns = target_wvl)
    
    var_start = True
    k = 0
    for iteration in range(n_iterations):
        XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.3, random_state=iteration)
        
        pls = PLSRegression(n_components=n_components)
        pls.fit(XX_train, yy_train)
        with open(f'{out_path}saved_models/{tr}_PLSR_model_interation{iteration+1}.pkl', 'wb') as f:
            pickle.dump(pls, f)
            
        coef = pls.coef_
        vvv = vip(XX_train, yy_train, pls)
        plsr_coef.iloc[k] = coef
        vip_score.iloc[k] = vvv
        
        pred = pls.predict(X_test)
        pred = pd.DataFrame(pred,columns = [f'iteration_{iteration+1}'])
        
        if var_start:
            iterative_pred = pred
            var_start = False
        else:
            iterative_pred = pd.concat([iterative_pred,pred], axis = 1)
        k = k+1

    y_test = pd.DataFrame(y_test, columns = [tr])
    df = pd.concat([y_test,iterative_pred], axis = 1)
    df["mean"] = iterative_pred.mean(axis = 1)
    df["std"] = iterative_pred.std(axis = 1)
    
    prefix = f"{out_path}{tr}_random CV_"
    df.to_csv(f"{prefix}df.csv", index=False)
    vip_score.to_csv(f"{prefix}VIP.csv", index=False)
    plsr_coef.to_csv(f"{prefix}coefficients.csv", index=False)
    press_scores.to_csv(f"{prefix}press.csv", index=False)
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

data_path = "/mnt/cephfs/scratch/groups/chen_group/FujiangJi/SmallSat_data/2_high_resolution_trait_maps/5_extract_training_samples/3_extracted_points.csv"
out_path = "/mnt/cephfs/scratch/groups/chen_group/FujiangJi/SmallSat_data/4_random_modeling_results/" 

models = ["EMIT", "Planet", "DSSFNET", "MSHFNET", "MSAHFNET", "SSFCNN", "TFNET", "CONSSFCNN", "RESTFNET", "MSDCNN", "SSRNET"]

traits = ['LWC_area', 'phosphorus', 'hemicellulose', 'cellulose', 'chl_area', 'phenolics_mg_g', 'LWC','potassium', 'sulfur', 'NSC_DS', 'nitrogen', 'lignin', 'LMA', '%C']

df = pd.read_csv(data_path)

for arch in models:    
    print(arch)
    os.makedirs(f"{out_path}/{arch}", exist_ok=True)
    export_path = f"{out_path}/{arch}/"
    os.makedirs(f"{export_path}/saved_models", exist_ok=True)
    #*********************************Extract training data*********************************#
    if arch == "SSFCNN":
        cols = [x for x in df.columns if arch in x and "CON" not in x]
    elif arch == "TFNET":
        cols = [x for x in df.columns if arch in x and "RES" not in x]
    else:
        cols = [x for x in df.columns if arch in x]
        
    refl = df[cols]
    if arch != "Planet":
        cols_to_drop = refl.columns[exclude_indices]
        refl = refl.drop(columns=cols_to_drop)

    tr_type = "upscaled" if arch == "EMIT" else "HR"
    tr_cols = [x for x in df.columns if tr_type in x]
    traits_df = df[tr_cols]
    
    for tr in traits:
        data =  pd.concat([refl, traits_df[[f"{tr_type}_{tr}"]]], axis = 1)
        data.dropna(axis=1, how = "all", inplace = True)
        data.dropna(axis=0, inplace = True)
        
        # data = data.iloc[0:200, :]             #*********************For Testing code*********************#
        X = data.iloc[:,:-1].values
        y = data.iloc[:,-1].values
        
        n_iterations = 20
        # n_iterations = 2
        if arch != "Planet":
            random_CV(X,y,tr,n_iterations, target_wvl, export_path)
        else:
            random_CV(X,y,tr,n_iterations, [443, 490, 531, 565, 610, 665, 705, 865], export_path)
