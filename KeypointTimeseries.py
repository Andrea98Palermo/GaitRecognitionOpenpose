import types
import sys


class KeypointTimeseries():
    timeseries = dict()

    """Result for BODY_25(25 body parts consisting of COCO + foot)
    const std::map < unsigned int, std::string > POSE_BODY_25_BODY_PARTS
    {{0, "Nose"},       {1, "Neck"},        {2, "RShoulder"},   {3, "RElbow"},      {4, "RWrist"},
    {5, "LShoulder"},   {6, "LElbow"},      {7, "LWrist"},      {8, "MidHip"},      {9, "RHip"},
    {10, "RKnee"},      {11, "RAnkle"},     {12, "LHip"},       {13, "LKnee"},      {14, "LAnkle"},
    {15, "REye"},       {16, "LEye"},       {17, "REar"},       {18, "LEar"},       {19, "LBigToe"},
    {20, "LSmallToe"},  {21, "LHeel"},      {22, "RBigToe"},    {23, "RSmallToe"},  {24, "RHeel"},
    {25, "Background"}}"""

    def __init__(self, args=None):
        for i in range(0, 26):
            self.timeseries[i] = list()

        """
        self.timeseries["Nose"] = list()
        self.timeseries["Neck"] = list()
        self.timeseries["RShoulder"] = list()
        self.timeseries["RElbow"] = list()
        self.timeseries["RWrist"] = list()
        self.timeseries["LShoulder"] = list()
        self.timeseries["LElbow"] = list()
        self.timeseries["LWrist"] = list()
        self.timeseries["MidHip"] = list()
        self.timeseries["RHip"] = list()
        self.timeseries["RKnee"] = list()
        self.timeseries["RAnkle"] = list()
        self.timeseries["LHip"] = list()
        self.timeseries["LKnee"] = list()
        self.timeseries["LAnkle"] = list()
        self.timeseries["REye"] = list()
        self.timeseries["LEye"] = list()
        self.timeseries["REar"] = list()
        self.timeseries["LEar"] = list()
        self.timeseries["LBigToe"] = list()
        self.timeseries["LSmallToe"] = list()
        self.timeseries["LHeel"] = list()
        self.timeseries["RBigToe"] = list()
        self.timeseries["RSmallToe"] = list()
        self.timeseries["RHeel"] = list()
        self.timeseries["Background"] = list()
        """

    def __getattr__(self, item):
        return self.timeseries



