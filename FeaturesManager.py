import os
import sys
sys.path.append(os.getcwd())
from KeypointsManager import KeypointsLoader

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
#from tsfresh import extract_features


class FeaturesManager():

    def __init__(self, from_csv = True):
        self.createFeaturesDF(from_csv=from_csv)
        self.cleanDataset()
        self.scaleDataset()
    
    #Creates features dataset from csv file or dictionary of timeseries
    def createFeaturesDF(self, columns=range(80), from_csv=True, csv_path="features"+os.sep+"features_flip.csv"):
        if from_csv:
                self.loadFromCSV(csv_path)
        else:
            joints, ids = KeypointsLoader().getDataset()
            all_walks = pd.DataFrame()
            nextIds = {id:0 for id in ids}

            for i, walk in enumerate(joints):
                walk_df = pd.DataFrame.from_dict(walk)[[str(c) for c in columns]]
                walk_df["walk_id"] = ids[i] + "_" + str(nextIds[ids[i]])
                nextIds[ids[i]] += 1
                for c in range(25):
                    try:
                        walk_df[[str(c)+'_1', str(c)+'_2']] = pd.DataFrame(walk_df[str(c)].tolist(), index=walk_df.index)
                        walk_df = walk_df.drop(str(c), axis="columns")
                    except Exception as e:
                        print(e)
                all_walks = pd.concat([all_walks, walk_df])
            self.extractFeatures(all_walks)

    def loadFromCSV(self, path):
        df = pd.read_csv(path).rename({'Unnamed: 0': 'walk_id'}, axis=1)
        self.ids = df['walk_id'].apply(lambda s: s[:3])
        self.features = df.drop("walk_id", axis='columns')

    def extractFeatures(self, df):
        #self.features = extract_features(df, column_id="walk_id")
        return

    def cleanDataset(self):
        self.features = self.features.replace([np.inf, -np.inf], np.nan)
        na_columns = [c for c in self.features.columns if self.features[c].isna().sum() >= 40]
        self.features = self.features.drop(na_columns, axis='columns')
        mean_vals = self.features.mean().fillna(self.features.mean().mean())
        self.features = self.features.fillna(mean_vals)

    def scaleDataset(self):
        constants = [c for c in self.features.columns if self.features[c].nunique() <= 1]
        self.features = self.features.drop(constants, axis='columns')
        numerical = [c for c in self.features.columns if self.features[c].nunique() > 2]
        self.features[numerical] = RobustScaler().fit_transform(self.features[numerical])
    
    def getDataset(self):
        return self.features, self.ids
