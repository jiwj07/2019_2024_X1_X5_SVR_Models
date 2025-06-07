# SVR X1-X5 2019-2024

# importing modules and packages
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from itertools import permutations
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import DictionaryLearning, FactorAnalysis, PCA, SparsePCA, IncrementalPCA, FastICA, TruncatedSVD, KernelPCA
from sklearn.feature_selection import SelectKBest, RFE, chi2, f_regression, mutual_info_regression, r_regression
from sklearn.linear_model import LinearRegression, Perceptron, LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import NuSVR, SVR
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# set print options
pd.set_option("display.max_rows", 10000)
pd.set_option("display.expand_frame_repr", True)
pd.set_option('display.width', 1000)

# importing data
df = pd.read_csv('2019-2024 River Data.csv')
print("2019-2024 River Data.csv")
print("Nonlinear SVM Models MSEs and MAEs")

# creating feature variables
X = df.drop(columns = ['Country', 'River', 'Land Area', 'Y AQI'])
y = df['Y AQI']

# creating train and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
results = []

# initialize models, scaler and KFold
models = {
    "NSVR-2": NuSVR(kernel='poly', degree=2),
    "NSVR-3": NuSVR(kernel='poly', degree=3),
    "NSVR-4": NuSVR(kernel='poly', degree=4),
    "NSVR-5": NuSVR(kernel='poly', degree=5),
    "NSVR-6": NuSVR(kernel='poly', degree=6),
    "NSVR-R": NuSVR(kernel='rbf'),
    "NSVR-S": NuSVR(kernel='sigmoid'),
    "ESVR-2": SVR(kernel='poly', degree=2),
    "ESVR-3": SVR(kernel='poly', degree=3),
    "ESVR-4": SVR(kernel='poly', degree=4),
    "ESVR-5": SVR(kernel='poly', degree=5),
    "ESVR-6": SVR(kernel='poly', degree=6),
    "ESVR-R": SVR(kernel='rbf'),
    "ESVR-S": SVR(kernel='sigmoid')
}
scaler = MinMaxScaler(feature_range=(0, 1))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for (model_name, model) in models.items():
    # creating train and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    # store MSE and MAE
    mse_scores = []
    mae_scores = []
    
    # K-fold cross validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
            X_train, X_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
            y_train, y_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
    
            # data normalization
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # predict on validation set
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            
            # calculate MAE and MSE
            mse = mean_squared_error(y_val, y_pred)    
            mae = mean_absolute_error(y_val, y_pred)
            mse_scores.append(mse)
            mae_scores.append(mae)
    
    # standardizing dataset
    X_temp_scaled = scaler.fit_transform(X_temp)
    X_test_scaled = scaler.transform(X_test)
    
    # train model
    model.fit(X_temp_scaled, y_temp)
    
    # predictions
    y_pred = model.predict(X_test_scaled)
    
    # compute MSE and MAE
    mse_cv_mean = np.mean(mse_scores)
    mse_cv_sd = np.std(mse_scores)
    mae_cv_mean = np.mean(mae_scores)
    mae_cv_sd = np.std(mae_scores)
    mse_final = mean_squared_error(y_test, y_pred)
    mae_final = mean_absolute_error(y_test, y_pred)
    results.append((model_name, "None", "None", mse_cv_mean, mse_cv_sd, mae_cv_mean, mae_cv_sd, mse_final, mae_final))
    
    
    # feature selection
    feature_selection_methods = {
        "ANOVA1": SelectKBest(score_func=f_regression, k=1),
        "ANOVA2": SelectKBest(score_func=f_regression, k=2),
        "ANOVA3": SelectKBest(score_func=f_regression, k=3),
        "CHISQ1": SelectKBest(score_func=chi2, k=1),
        "CHISQ2": SelectKBest(score_func=chi2, k=2),
        "CHISQ3": SelectKBest(score_func=chi2, k=3),
        "MIT1": SelectKBest(score_func=mutual_info_regression, k=1),
        "MIT2": SelectKBest(score_func=mutual_info_regression, k=2),
        "MIT3": SelectKBest(score_func=mutual_info_regression, k=3),
        "PCR1": SelectKBest(score_func=r_regression, k=1),
        "PCR2": SelectKBest(score_func=r_regression, k=2),
        "PCR3": SelectKBest(score_func=r_regression, k=3),
        # "RFE1": RFE(model, n_features_to_select=1),
        # "RFE2": RFE(model, n_features_to_select=2),
        # "RFE3": RFE(model, n_features_to_select=3),
    }
    
    for (fs_name, fs) in feature_selection_methods.items():
    
        # store MSE and MAE
        mse_scores_fs = []
        mae_scores_fs = []
        
        # K-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
            X_train, X_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
            y_train, y_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
    
            # data normalization
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # apply feature selection on train and validation set
            X_train_fs = fs.fit_transform(X_train_scaled, y_train)
            X_val_fs = fs.transform(X_val_scaled)
            
            # predict on validation set
            model.fit(X_train_fs, y_train)
            y_pred = model.predict(X_val_fs)
            
            # calculate MAE and MSE
            mse = mean_squared_error(y_val, y_pred)    
            mae = mean_absolute_error(y_val, y_pred)
            mse_scores_fs.append(mse)
            mae_scores_fs.append(mae)
            
        # standardizing dataset
        X_temp_scaled = scaler.fit_transform(X_temp)
        X_test_scaled = scaler.transform(X_test)
    
        # apply feature selection on test set
        X_temp_fs = fs.fit_transform(X_temp_scaled, y_temp)
        X_test_fs = fs.transform(X_test_scaled)
    
        # train model
        model.fit(X_temp_fs, y_temp)
        
        # predictions
        y_pred = model.predict(X_test_fs)
        
        # compute MSE and MAE
        mse_cv_mean = np.mean(mse_scores_fs)
        mse_cv_sd = np.std(mse_scores_fs)
        mae_cv_mean = np.mean(mae_scores_fs)
        mae_cv_sd = np.std(mae_scores_fs)
        mse_final = mean_squared_error(y_test, y_pred)
        mae_final = mean_absolute_error(y_test, y_pred)
        results.append((model_name, fs_name, "None", mse_cv_mean, mse_cv_sd, mae_cv_mean, mae_cv_sd, mse_final, mae_final))
    
    # dimensionality reduction
    dimensionality_reduction_techniques = {
        "DL1": DictionaryLearning(n_components=1, max_iter=1000),
        "DL2": DictionaryLearning(n_components=2, max_iter=1000),
        "DL3": DictionaryLearning(n_components=3, max_iter=1000),
        "FA1": FactorAnalysis(n_components=1, max_iter=1000),
        "FA2": FactorAnalysis(n_components=2, max_iter=1000),
        "FA3": FactorAnalysis(n_components=3, max_iter=1000),
        "ICA1": FastICA(n_components=1, tol=0.05, max_iter=1000),
        "ICA2": FastICA(n_components=2, tol=0.05, max_iter=1000),
        "ICA3": FastICA(n_components=3, tol=0.05, max_iter=1000),
        "IPCA1": IncrementalPCA(n_components=1),
        "IPCA2": IncrementalPCA(n_components=2),
        "IPCA3": IncrementalPCA(n_components=3),
        "KPCA1-2": KernelPCA(n_components=1, kernel='poly', degree=2),
        "KPCA2-2": KernelPCA(n_components=2, kernel='poly', degree=2),
        "KPCA3-2": KernelPCA(n_components=3, kernel='poly', degree=2),
        "KPCA1-3": KernelPCA(n_components=1, kernel='poly', degree=3),
        "KPCA2-3": KernelPCA(n_components=2, kernel='poly', degree=3),
        "KPCA3-3": KernelPCA(n_components=3, kernel='poly', degree=3),
        "KPCA1-4": KernelPCA(n_components=1, kernel='poly', degree=4),
        "KPCA2-4": KernelPCA(n_components=2, kernel='poly', degree=4),
        "KPCA3-4": KernelPCA(n_components=3, kernel='poly', degree=4),
        "KPCA1-5": KernelPCA(n_components=1, kernel='poly', degree=5),
        "KPCA2-5": KernelPCA(n_components=2, kernel='poly', degree=5),
        "KPCA3-5": KernelPCA(n_components=3, kernel='poly', degree=5),
        "KPCA1-6": KernelPCA(n_components=1, kernel='poly', degree=6),
        "KPCA2-6": KernelPCA(n_components=2, kernel='poly', degree=6),
        "KPCA3-6": KernelPCA(n_components=3, kernel='poly', degree=6),
        "KPCA1-R": KernelPCA(n_components=1, kernel='rbf'),
        "KPCA2-R": KernelPCA(n_components=2, kernel='rbf'),
        "KPCA3-R": KernelPCA(n_components=3, kernel='rbf'),
        "KPCA1-S": KernelPCA(n_components=1, kernel='sigmoid'),
        "KPCA2-S": KernelPCA(n_components=2, kernel='sigmoid'),
        "KPCA3-S": KernelPCA(n_components=3, kernel='sigmoid'),
        "KPCA1-C": KernelPCA(n_components=1, kernel='cosine'),
        "KPCA2-C": KernelPCA(n_components=2, kernel='cosine'),
        "KPCA3-C": KernelPCA(n_components=3, kernel='cosine'),
        "LPCA1": PCA(n_components=1),
        "LPCA2": PCA(n_components=2),
        "LPCA3": PCA(n_components=3),
        "SPCA1": SparsePCA(n_components=1),
        "SPCA2": SparsePCA(n_components=2),
        "SPCA3": SparsePCA(n_components=3),
        "TSVD1": TruncatedSVD(n_components=1), 
        "TSVD2": TruncatedSVD(n_components=2), 
        "TSVD3": TruncatedSVD(n_components=3)
    }
    
    for (dr_name, dr) in dimensionality_reduction_techniques.items():
        
        # store MSE and MAE
        mse_scores_dr = []
        mae_scores_dr = []
        
        # K-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
            X_train, X_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
            y_train, y_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
    
            # data normalization
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # apply feature selection on train and validation set
            X_train_dr = dr.fit_transform(X_train_scaled, y_train)
            X_val_dr = dr.transform(X_val_scaled)
            
            # predict on validation set
            model.fit(X_train_dr, y_train)
            y_pred = model.predict(X_val_dr)
            
            # calculate MAE and MSE
            mse = mean_squared_error(y_val, y_pred)    
            mae = mean_absolute_error(y_val, y_pred)
            mse_scores_dr.append(mse)
            mae_scores_dr.append(mae)
            
        # standardizing dataset
        X_temp_scaled = scaler.fit_transform(X_temp)
        X_test_scaled = scaler.transform(X_test)
    
        # apply feature selection on test set
        X_temp_dr = dr.fit_transform(X_temp_scaled, y_temp)
        X_test_dr = dr.transform(X_test_scaled)
    
        # train model
        model.fit(X_temp_dr, y_temp)
        
        # predictions
        y_pred = model.predict(X_test_dr)
        
        # compute MSE
        mse_cv_mean = np.mean(mse_scores_dr)
        mse_cv_sd = np.std(mse_scores_dr)
        mae_cv_mean = np.mean(mae_scores_dr)
        mae_cv_sd = np.std(mae_scores_dr)
        mse_final = mean_squared_error(y_test, y_pred)
        mae_final = mean_absolute_error(y_test, y_pred)
        results.append((model_name, "None", dr_name, mse_cv_mean, mse_cv_sd, mae_cv_mean, mae_cv_sd, mse_final, mae_final))
    
    
    # 2-layer
    for (fs_name, fs) in feature_selection_methods.items():
        
        for(dr_name, dr) in dimensionality_reduction_techniques.items(): 
            if (fs_name == "ANOVA1" or fs_name == "CHISQ1" or fs_name == "MIT1" or fs_name == "PCR1" or fs_name == "RFE1"): 
                continue
            if (dr_name == "DL3" or dr_name == "FA3" or dr_name == "ICA3" or dr_name == "IPCA3" or dr_name == "KPCA3-1" or dr_name == "KPCA3-2" or dr_name == "KPCA3-3" or dr_name == "KPCA3-4" or dr_name == "KPCA3-5" or dr_name == "KPCA3-6" or dr_name == "KPCA3-R" or dr_name == "KPCA3-S" or dr_name == "KPCA3-C" or dr_name == "LPCA3" or dr_name == "SPCA3" or dr_name == "TSVD3"):
                continue
            if (fs_name == "ANOVA2" or fs_name == "CHISQ2" or fs_name == "MIT2" or fs_name == "PCR2" or fs_name == "RFE2") and (dr_name == "DL2" or dr_name == "FA2" or dr_name == "KPCA2-1" or dr_name == "KPCA2-2" or dr_name == "KPCA2-3" or dr_name == "KPCA2-4" or dr_name == "KPCA2-5" or dr_name == "KPCA2-6" or dr_name == "KPCA2-R" or dr_name == "KPCA2-S" or dr_name == "KPCA2-C" or dr_name == "LPCA2" or dr_name == "SPCA2" or dr_name == "TSVD2"):
                continue
            
            # store MSE and MAE
            mse_scores_fs_dr = []
            mae_scores_fs_dr = []
            
            # K-fold cross validation
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_temp)):
                X_train, X_val = X_temp.iloc[train_idx], X_temp.iloc[val_idx]
                y_train, y_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
        
                # data normalization
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # apply feature selection on train and validation set
                X_train_fs = fs.fit_transform(X_train_scaled, y_train)
                X_val_fs = fs.transform(X_val_scaled)
    
                # apply dimsenionality reduction on train and validation set
                X_train_dr = dr.fit_transform(X_train_fs, y_train)
                X_val_dr = dr.transform(X_val_fs)
                
                # predict on validation set
                model.fit(X_train_dr, y_train)
                y_pred = model.predict(X_val_dr)
                
                # calculate MAE and MSE
                mse = mean_squared_error(y_val, y_pred)    
                mae = mean_absolute_error(y_val, y_pred)
                mse_scores_fs_dr.append(mse)
                mae_scores_fs_dr.append(mae)
                
            # standardizing dataset
            X_temp_scaled = scaler.fit_transform(X_temp)
            X_test_scaled = scaler.transform(X_test)
        
            # apply feature selection
            X_temp_fs = fs.fit_transform(X_temp_scaled, y_temp)
            X_test_fs = fs.transform(X_test_scaled)
    
            # apply dimensionality reduction
            X_temp_dr = dr.fit_transform(X_temp_fs, y_temp)
            X_test_dr = dr.transform(X_test_fs)
        
            # train model
            model.fit(X_temp_dr, y_temp)
            
            # predictions
            y_pred = model.predict(X_test_dr)
            
            # compute MSE and MAE
            mse_cv_mean = np.mean(mse_scores_fs_dr)
            mse_cv_sd = np.std(mse_scores_fs_dr)
            mae_cv_mean = np.mean(mae_scores_fs_dr)
            mae_cv_sd = np.std(mae_scores_fs_dr)
            mse_final = mean_squared_error(y_test, y_pred)
            mae_final = mean_absolute_error(y_test, y_pred)
            results.append((model_name, fs_name, dr_name, mse_cv_mean, mse_cv_sd, mae_cv_mean, mae_cv_sd, mse_final, mae_final))


# print results
print()
df_results = pd.DataFrame(results, columns=["Model", "Feature_Selection", "Dimensionality_Reduction_Technique", "CV MSE Mean", "CV MSE SD", "CV MAE Mean", "CV MAE SD", "Final MSE", "Final MAE"])
print(df_results)
df_results.to_csv('2019-2024 X1-X5 Results SVM KFold.csv', index=False)
