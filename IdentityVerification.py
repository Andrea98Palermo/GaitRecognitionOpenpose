import os
import sys
sys.path.append(os.getcwd())
from FeaturesManager import FeaturesManager

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold, SelectPercentile, SelectKBest, SelectFromModel, chi2, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from collections import defaultdict


class DatasetHandler:

    def __init__(self, X, y, val_size=0.2, test_size=0.2, seed=42):
        self.splitData(X, y, val_size, test_size, seed)
        self.X_train_selected, self.X_val_selected, self.X_test_selected = self.X_train.to_numpy(), self.X_val.to_numpy(), self.X_test.to_numpy()

    #splits dataset with 60-20-20 proportion
    def splitData(self, X, y, val_size, test_size, seed):
        self.X_train, X_test, self.y_train, y_test = train_test_split(X, y, test_size=val_size+test_size, random_state=seed, stratify=y)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_test, y_test, test_size=test_size/(val_size+test_size), random_state=seed, stratify=y_test)

    def performSelectionStep(self, selector):       #selector has to be fitted
        self.X_train_selected = selector.transform(self.X_train_selected)
        self.X_val_selected = selector.transform(self.X_val_selected)
        self.X_test_selected = selector.transform(self.X_test_selected)

    #Resets X to all original features
    def resetSelection(self):
        self.X_train_selected, self.X_val_selected, self.X_test_selected = self.X_train.to_numpy(), self.X_val.to_numpy(), self.X_test.to_numpy()


#Principal Feature Analysis
class PFA():

    def __init__(self, n_features, q=None, seed=42):
        self.q = q
        self.n_features = n_features
        self.seed = seed

    def fit(self, X, y=None):
        if not self.q:
            self.q = X.shape[1]

        pca = PCA(n_components=self.q, random_state=self.seed).fit(X)                        
        A_q = pca.components_.T                                                         

        kmeans = KMeans(n_clusters=self.n_features, random_state=self.seed).fit(A_q)        
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_                                      

        dists = defaultdict(list)
        for i, c in enumerate(clusters):                                                
            dist = metrics.pairwise.euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]

    def transform(self, X):
        return X[:, self.indices_]

class FeaturesSelector:

    def __init__(self):
        X, y = FeaturesManager().getDataset()
        self.dataset = DatasetHandler(X, y)
        self.selectFeatures(VarianceThreshold(0.0))

    #Performs selection with given (unfitted) selector
    def selectFeatures(self, selector):
        X = np.concatenate((self.dataset.X_train_selected, self.dataset.X_val_selected), axis=0)
        y = pd.concat([self.dataset.y_train, self.dataset.y_val])
        selector.fit(X, y)
        self.dataset.performSelectionStep(selector)

    #Computes results on validation set with the given range of feature numbers
    def validateSelection(n_features_range=range(1, 100)):
        v = IdentityVerifier()
        for k in n_features_range:
            print("Results using " + k + " features:")
            try:
                v.selector.selectFeatures(SelectKBest(k=8))
                v.validateAllModels({"C": 0.15, "kernel":"rbf", "gamma":"auto", "coef0":0.0})
                v.selector.resetSelection()
            except Exception as e:
                print(e)

    #Resets X to all original features
    def resetSelection(self):
        self.dataset.resetSelection()

    def getDataset(self):
        return self.dataset


class IdentityVerifier():

    def __init__(self, seed=42):
        self.selector = FeaturesSelector()
        self.seed = seed
        self.models = {}

    #Trains model for a single subject
    def trainPersonalModel(self, subjectId, clf=SVC(probability=True, random_state=42), oversampling=True):
        dataset = self.selector.getDataset()
        y_train_bin = [1 if y == subjectId else 0 for y in dataset.y_train] 
        if oversampling:
            X_train, y_train = self.performOverSampling(y_train_bin)
        else:
            X_train, y_train = dataset.X_train_selected.copy(), y_train_bin

        clf.fit(X_train, y_train)
        self.models[subjectId] = clf

    def trainAllModels(self, params):
        for id in set(self.selector.getDataset().y_train):
            clf = SVC(probability=True, random_state=self.seed, C=params["C"], kernel=params["kernel"], gamma=params["gamma"], coef0=params["coef0"])
            self.trainPersonalModel(id, clf)

    def performOverSampling(self, y_bin):
        sm = SMOTE(random_state=self.seed)                                                     
        X_train_upsampled, y_train_upsampled = sm.fit_resample(self.selector.getDataset().X_train_selected, y_bin)        #upsampling classe minoritaria
        return X_train_upsampled, y_train_upsampled

    #Returns predictions of a single subject's model for test set
    def testModel(self, subjectId):
        dataset = self.selector.getDataset()
        y_test_bin = [1 if y == subjectId else 0 for y in dataset.y_test] 
        y_scores = self.models[subjectId].predict_proba(dataset.X_test_selected)               
        y_genuine_scores = [x[1] for x in y_scores]
        return y_test_bin, y_genuine_scores
    
    #computes EER, ROC, ZeroFAR for all models with test set
    def testAllModels(self, params, ZeroFAR=False):
        self.trainAllModels(params)

        y_test_bin_all = []
        y_scores_all = []
        for subjectId in self.models.keys():
            y_test_bin, y_genuine_scores = self.testModel(subjectId)
            y_test_bin_all.extend(y_test_bin)               
            y_scores_all.extend(y_genuine_scores)           
        
        FARs, FRRs, GARs, HTERs, treshs = IdentityVerifier.computeRates(y_scores_all, y_test_bin_all)
        IdentityVerifier.computeEER(FARs, FRRs, treshs)
        IdentityVerifier.ComputeROC(FARs, GARs)
        if ZeroFAR:
            IdentityVerifier.computeZeroFAR(FARs, FRRs)
    
    #Returns predictions of a single subject's model for validation set
    def validateModel(self, subjectId):
        dataset = self.selector.getDataset()
        y_test_bin = [1 if y == subjectId else 0 for y in dataset.y_val] 
        y_scores = self.models[subjectId].predict_proba(dataset.X_val_selected)               
        y_genuine_scores = [x[1] for x in y_scores]
        return y_test_bin, y_genuine_scores
    
    #computes EER, ROC, ZeroFAR for all models with validation set
    def validateAllModels(self, params, ZeroFAR=False):
        self.trainAllModels(params) 

        y_test_bin_all = []
        y_scores_all = []
        for subjectId in self.models.keys():
            y_test_bin, y_genuine_scores = self.validateModel(subjectId)
            y_test_bin_all.extend(y_test_bin)               
            y_scores_all.extend(y_genuine_scores)           
        
        FARs, FRRs, GARs, HTERs, treshs = IdentityVerifier.computeRates(y_scores_all, y_test_bin_all)
        IdentityVerifier.computeEER(FARs, FRRs, treshs)
        IdentityVerifier.ComputeROC(FARs, GARs)
        if ZeroFAR:
            IdentityVerifier.computeZeroFAR(FARs, FRRs)
    
    #tunes parameters from given grid 
    def tuneModelsParameters(self, params_grid):
        for C in params_grid["C"]:
            for kernel in params_grid["kernel"]:
                for gamma in params_grid["gamma"]:
                    for coef0 in params_grid["coef0"]:
                        print("C: ", C, " kernel: ", kernel, " gamma :", gamma, " coef0: ", coef0)
                        self.validateAllModels({"C":C, "kernel":kernel, "gamma":gamma, "coef0":coef0})

    #Returns FARs, FRRs, GARs, HTERs and tresholds given the predictions of all models
    def computeRates(y_scores_all, y_test_bin_all):
        FARs = []
        FRRs = []
        GARs = []
        HTERs = []
        treshs = np.linspace(0,1,101)              
        for t in treshs:
            y_pred_t = [1 if x >= t else 0 for x in y_scores_all]                                           
            tn, fp, fn, tp = metrics.confusion_matrix(y_test_bin_all, y_pred_t, labels=[0,1]).ravel()       
            FAR = fp/(fp + tn)                                                                              
            FRR = fn/(fn + tp)
            GAR = 1 - FRR
            HTER = (FAR + FRR)/2 
            FARs = np.append(FARs, FAR)                                                                     
            FRRs = np.append(FRRs, FRR)
            GARs = np.append(GARs, GAR)
            HTERs = np.append(HTERs, HTER)
        return FARs, FRRs, GARs, HTERs, treshs

    #computes EER and optionally plots FAR and FRR curves
    def computeEER(FARs, FRRs, treshs, plot=False):
        abs_diffs = np.abs(FARs - FRRs)             
        min_index = np.argmin(abs_diffs)
        eer = np.mean((FARs[min_index], FRRs[min_index]))
        threshold = treshs[min_index]
        print("EER:------------>", eer)
        print("treshold:------------>", threshold)
        if plot:
            plt.plot(FARs, label="FAR", color="#fc03e8", linewidth=1)
            plt.plot(FRRs, label="FRR", linewidth=1)
            plt.margins(0.01)
            plt.xlabel("Treshold")
            plt.ylabel("Error rate")
            plt.legend()
            plt.show()

    def computeZeroFAR(FARs, FRRs):
        #computing zeroFAR (<0.001)
        try:
            zeroFAR_index = np.where(FARs < 0.001)[0][0]
            zeroFAR = FRRs[zeroFAR_index]
            print("ZeroFAR_0.001:------------>", zeroFAR)
        except:
            print("Nessun valore FAR < 0.001")

        #computing zeroFAR (<0.0001)
        try:
            zeroFAR_index = np.where(FARs < 0.0001)[0][0]
            zeroFAR = FRRs[zeroFAR_index]
            print("ZeroFAR_0.0001:------------>", zeroFAR)
        except:
            print("Nessun valore FAR < 0.0001")
    
    #computes AUC and optionally plots ROC
    def ComputeROC(FARs, GARs, plot=False):
        auc = trapz(GARs, FARs)
        print("AUC = ", auc)
        if plot:
            plt.plot(FARs, GARs, 'r-', label="ROC", linewidth=3)
            plt.xlabel("FAR")
            plt.ylabel("1-FRR(GAR)")
            plt.legend()
            plt.show()

if __name__ == "__main__":
    
    v = IdentityVerifier()
    v.selector.selectFeatures(SelectKBest(k=8))
    models_params = {"C": 0.15, "kernel":"rbf", "gamma":"auto", "coef0":0.0}    #optimal parameters found with tuning
    v.testAllModels(models_params)
    