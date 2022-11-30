import os
import numpy as np
from scipy.stats import entropy

class KeypointsLoader():

    def __init__(self, baseDir="joints/extractedJoints"):
        self.dataset = list()
        self.ids = list()
        for walk in os.listdir(baseDir):
            self.dataset.append(self.openKeypointsFile(baseDir + os.sep + walk))

    def openKeypointsFile(self, path):
        with open(path, "r") as k:
            r = k.read()
            timeseries = dict()
            r = r.split("\n")
            self.ids.append(r[0])
            index = 1
            while index < len(r) - 1:
                lst = r[index].split(":")
                timeseries[lst[0]] = eval(lst[1])
                index += 1
        return timeseries

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return ''.join(str(a) + "\n" + str(b) + "\n" for (a, b) in zip(self.ids, self.dataset))

    def getDataset(self):
        return self.dataset, self.ids

def test():
    app = KeypointsLoader()
    print(len(app))
    print(app)

def statistics():
    app=KeypointsLoader()
    datasetApp = []
    for elem in app.dataset:
        app = [x for x in elem.values()]
        walk = []
        for i in range(25):
            walk.append([y[0] for y in app[i]])
            walk.append([y[1] for y in app[i]])
        for i in range(26, 30):
            walk.append(app[i])
        for i in range(30, 55):
            walk.append(app[i])
        for i in range(55, 80):
            walk.append(app[i])
        datasetApp.append(walk)
    """stds=[]
    for elem in datasetApp:
        walk = []
        for i in elem:
            walk.append(np.std(i))
        stds.append(np.array(walk))
    stds=np.array(stds)
    meanSTDs=[]
    for j in range(len(stds[0])):
        meanSTDs.append(np.mean(stds[:,[j]]))
    for i in range(len(meanSTDs)):
        print(str(i)+" : "+str(meanSTDs[i]))"""

    entropies = []
    for elem in datasetApp:
        walk = []
        for i in elem:
            walk.append(entropy(i))
        entropies.append(np.array(walk))
    stds = np.array(entropies)
    meanEntropies = []
    for j in range(len(stds[0])):
        meanEntropies.append(np.mean(stds[:, [j]]))
    for i in range(len(meanEntropies)):
        print(str(i) + " : " + str(meanEntropies[i]))

if __name__ == "__main__":
    statistics()
    #test()

"""class KeypointsLoader():

    def openKeypointsFile(self, path):
        with open(path, "r") as k:
            r = k.read()
            timeseries = dict()
            r = r.replace('{', '')
            r = r.replace('}', '')
            lst = r.split(":")
            index = 0
            while index < len(lst) - 1:
                # key=lst[int(index)]
                j = lst[index + 1].replace('[', '')
                j = j.replace(']', '')
                j = j.replace(' ', '')
                j = j.replace("),(", ' ')
                j = j.replace(')', '')
                j = j.replace('(', '')
                coords = j.split()
                keypoints = list()
                if index < 25:
                    for i in coords:
                        app = i.split(',')
                        keypoints.append((int(app[0]), int(app[1])))
                else:
                    for i in coords[0].split(','):
                        if i != 'None':
                            keypoints.append(float(i))
                        else:
                            keypoints.append(None)
                timeseries[index] = keypoints
                # joints=list(lst[index+1])
                # timeseries[key]=joints
                index += 1
        return timeseries"""
"""
    timeseries=openKeypointsFile("extractedJoints\\fyc_00_1.txt")
    print(timeseries)

"""    """def loadDataset(self, baseDir="extractedJoints"):
        dataset = list()
        for walk in os.listdir(baseDir):
            dataset.append(KeypointsLoader.openKeypointsFile(self, baseDir + os.sep + walk))
        return dataset

def test():
    app = KeypointsLoader.loadDataset()
    print(len(app))
    min = len(app[0][12])
    max = len(app[0][12])
    avg = 0
    for i in range(0, len(app)):
        a = len(app[i][12])
        avg += a
        if min > a:
            min = a
            print(i)
        if max < a:
            max = a
    avg /= len(app)
    print(min)
    print(max)
    print(avg)

    # print(app)

"""
