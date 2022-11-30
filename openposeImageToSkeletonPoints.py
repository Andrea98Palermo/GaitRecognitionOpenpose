# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import math
import numpy as np

# from KeypointTimeseries import KeypointTimeseries

# timeseries of joint locations
timeseries = dict()

"""Result for BODY_25(25 body parts consisting of COCO + foot)
const std::map < unsigned int, std::string > POSE_BODY_25_BODY_PARTS
{{0, "Nose"},       {1, "Neck"},        {2, "RShoulder"},   {3, "RElbow"},      {4, "RWrist"},
{5, "LShoulder"},   {6, "LElbow"},      {7, "LWrist"},      {8, "MidHip"},      {9, "RHip"},
{10, "RKnee"},      {11, "RAnkle"},     {12, "LHip"},       {13, "LKnee"},      {14, "LAnkle"},
{15, "REye"},       {16, "LEye"},       {17, "REar"},       {18, "LEar"},       {19, "LBigToe"},
{20, "LSmallToe"},  {21, "LHeel"},      {22, "RBigToe"},    {23, "RSmallToe"},  {24, "RHeel"},
{25, "Background"}}"""


def initTimeseries(timeseries):
    for i in range(0, 26):
        timeseries[i] = list()


# timeseries dict(bodyLocationName - 0-25, normalized_coordinates - list(2 - (x,y)))
# normalized_coordinates=x-minX, y-minY
"""def imageToKeypoints(timeseries, datum):
    for i in datum.poseKeypoints:
        index = 0
        for j in i:
            timeseries[index].append((int(j[1]), int(j[0])))
            index += 1"""

"""def imageToKeypoints(timeseries, datum):
    for i in datum.poseKeypoints:
        index = 0
        minX = sys.maxsize - 1
        minY = sys.maxsize - 1

        for j in i:
            if int(j[1]) < minX and int(j[1]) > 0:
                minX = int(j[1])
            if int(j[0]) < minY and int(j[0]) > 0:
                minY = int(j[0])
        print("MINX"+str(minX))
        print("MINY"+str(minY))
        for j in i:
            timeseries[index].append((int(j[1]) - minX, int(j[0]) - minY))
            index += 1"""


# insert the datum into the timeseries
def imageToKeypoints(timeseries, datum):
    for i in datum.poseKeypoints:
        index = 0
        for j in i:
            timeseries[index].append((int(j[1]), int(j[0])))
            index += 1


# assign the previous/following value to the keypoints that are not correctly estimated
def fix0Keypoints(timeseries):
    #print(timeseries)
    for i in range(0, len(timeseries)):
        app: list = timeseries[i]
        repeat = True
        repeatCount = 0
        while repeat and repeatCount < len(app):
            repeat = False
            repeatCount += 1
            for j in range(0, len(app)):
                if app[j] == (0, 0):
                    repeat = True
                    if j == 0:
                        timeseries[i][j] = app[j + 1]
                    else:
                        timeseries[i][j] = app[j - 1]
    return timeseries


def normalizeKeypoints(timeseries):
    newTimeseries = dict()
    initTimeseries(newTimeseries)
    newTimeseries.pop(25)
    # for each time instant i.e. frame
    for timeInstant in range(0, len(timeseries[0])):
        # find minX and minY
        minX = sys.maxsize - 1
        minY = sys.maxsize - 1
        for keypoint in range(0, 25):
            if timeseries[keypoint][timeInstant][1] < minX:
                minX = timeseries[keypoint][timeInstant][1]
            if timeseries[keypoint][timeInstant][0] < minY:
                minY = timeseries[keypoint][timeInstant][0]
        # normalize keypoints
        for keypoint in range(0, len(timeseries.keys())):
            newTimeseries[keypoint].append((timeseries[keypoint][timeInstant][1] - minX, timeseries[keypoint][timeInstant][0] - minY))
    return newTimeseries


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


# derivative for velocity and acceleration
def line_len_1d(p1, p2):
    return np.sqrt(np.square(p2 - p1))


def line_len_2d(p1, p2):
    return np.sqrt(np.square(p2[0] - p1[0]) + np.square(p2[1] - p1[1]))


def computeExtraFeatures(normTimeseries):
    heights = list()
    leftLegAngles = list()
    rightLegAngles = list()
    leftArmAngles = list()
    rightArmAngles = list()
    for i in range(0, len(normTimeseries[0])):
        heights.append(
            max(normTimeseries[19][i][1], normTimeseries[20][i][1], normTimeseries[22][i][1], normTimeseries[23][i][1],
                normTimeseries[11][i][1], normTimeseries[14][i][1], normTimeseries[24][i][1], normTimeseries[21][i][1]))
        if normTimeseries[9][i] != (0, 0) and normTimeseries[10][i] != (0, 0) and normTimeseries[11][i] != (0, 0):
            leftLegAngles.append(round(getAngle(normTimeseries[9][i], normTimeseries[10][i], normTimeseries[11][i]), 2))
        else:
            leftLegAngles.append(-1)
        if normTimeseries[12][i] != (0, 0) and normTimeseries[13][i] != (0, 0) and normTimeseries[14][i] != (0, 0):
            rightLegAngles.append(
                round(getAngle(normTimeseries[12][i], normTimeseries[13][i], normTimeseries[14][i]), 2))
        else:
            leftLegAngles.append(-1)
        if normTimeseries[2][i] != (0, 0) and normTimeseries[3][i] != (0, 0) and normTimeseries[4][i] != (0, 0):
            leftArmAngles.append(round(getAngle(normTimeseries[2][i], normTimeseries[3][i], normTimeseries[4][i]), 2))
        else:
            leftArmAngles.append(-1)
        if normTimeseries[5][i] != (0, 0) and normTimeseries[6][i] != (0, 0) and normTimeseries[7][i] != (0, 0):
            rightArmAngles.append(round(getAngle(normTimeseries[5][i], normTimeseries[6][i], normTimeseries[7][i]), 2))
        else:
            rightArmAngles.append(-1)
    return heights, leftLegAngles, rightLegAngles, leftArmAngles, rightArmAngles
    """normTimeseries[25]=heights
    normTimeseries[26]=leftLegAngles
    normTimeseries[27]=rightLegAngles
    print(heights)
    print(leftLegAngles)
    print(rightLegAngles)"""


def computeVelocitiesAccelerations(timeseries, fps):
    velocities = dict()
    accelerations = dict()
    # compute velocities
    for keypoint in range(0, 25):
        velocity = list()
        for i in range(1, len(timeseries[keypoint])):
            velocity.append(line_len_2d(timeseries[keypoint][i - 1], timeseries[keypoint][i]) / fps)
            if i == 1:
                velocity.append(line_len_2d(timeseries[keypoint][i - 1], timeseries[keypoint][i]) / fps)
        velocities[keypoint] = velocity
    # compute accelerations
    for keypoint in range(0, 25):
        acceleration = list()
        for i in range(1, len(velocities[keypoint])):
            acceleration.append(line_len_1d(velocities[keypoint][i - 1], velocities[keypoint][i]) / fps)
            if i == 2:
                acceleration.append(line_len_1d(velocities[keypoint][i - 1], velocities[keypoint][i]) / fps)
        accelerations[keypoint] = acceleration
    return velocities, accelerations


"""def timeseriesNormalization(timeseries):
        minX_list=list()
        minY_list=list()
        for i in range(0, 25):
            minX_list.append(min(timeseries[i][0]))
            minY_list.append(min(timeseries[i][1]))
            print(minY_list[i])
        print("done")
        for i in range(0, 25):
            for j in range (0, len(timeseries[i])):
                print(timeseries[i][j][0])
                timeseries[i][j][0] -= minX_list[i]
                timeseries[i][j][1] -= minY_list[i]
        print("TODO")"""

# main
try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' + dir_path + '/../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    """
    # Flags
    parser = argparse.ArgumentParser()
    #parser.add_argument("--image_path", default="fyc-00_1-001.png", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../models"  # "../../../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item
            
        # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    op.PoseModel = op.PoseModel.BODY_25
    # poseModel = op.PoseModel.BODY_25
    # print(op.getPoseBodyPartMapping(poseModel))
    # print(op.getPoseNumberBodyParts(poseModel))
    # print(op.getPosePartPairs(poseModel))
    # print(op.getPoseMapIndex(poseModel))

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    """
    newDir = "D:\\OpenPose Build\\code\\extractedJointsAugmented"
    #newDir = "D:\\OpenPose Build\\code\\extractedJoints"
    try:
        os.mkdir(newDir)
    except OSError:
        None

    # dbRoot="C:\\Users\\aless\\Downloads\\GaitDB\\DatasetA-002\\DatasetA\\gaitdb"
    #dbRoot = "D:\\gaitdb"
    dbRoot = "D:\\gaitdbFlip"
    for username in os.listdir(dbRoot):
        for user in username:
            if user != "video" and user != "member_photo":
                userRoot = dbRoot + "\\" + username
                for walk in os.listdir(userRoot):
                    argDefault = userRoot + "\\" + walk
                    print(argDefault)
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--image_dir",
                                        # default="C:\\Users\\aless\\Downloads\\GaitDB\\DatasetA-002\\DatasetA\\gaitdb\\fyc\\00_p",
                                        default=argDefault,
                                        help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
                    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
                    args = parser.parse_known_args()

                    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
                    params = dict()
                    params["model_folder"] = "../models/"

                    # Add others in path?
                    for i in range(0, len(args[1])):
                        curr_item = args[1][i]
                        if i != len(args[1]) - 1:
                            next_item = args[1][i + 1]
                        else:
                            next_item = "1"
                        if "--" in curr_item and "--" in next_item:
                            key = curr_item.replace('-', '')
                            if key not in params:  params[key] = "1"
                        elif "--" in curr_item and "--" not in next_item:
                            key = curr_item.replace('-', '')
                            if key not in params: params[key] = next_item

                    # Construct it from system arguments
                    # op.init_argv(args[1])
                    # oppython = op.OpenposePython()

                    # Starting OpenPose
                    opWrapper = op.WrapperPython()
                    opWrapper.configure(params)
                    opWrapper.start()

                    # Read frames on directory
                    imagePaths = op.get_images_on_directory(args[0].image_dir);
                    timeseriesFile = newDir + "\\" + imagePaths[0].split("\\")[-3] + "_" + imagePaths[0].split("\\")[-2] + ".txt"
                    if True:  # not os.path.exists(timeseriesFile):
                        start = time.time()

                        op.PoseModel = op.PoseModel.BODY_25
                        initTimeseries(timeseries)

                        # Process images and convert into timeseries
                        for imagePath in imagePaths:
                            datum = op.Datum()
                            imageToProcess = cv2.imread(imagePath)
                            datum.cvInputData = imageToProcess
                            opWrapper.emplaceAndPop([datum])
                            imageToKeypoints(timeseries, datum)
                            """cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
                            cv2.waitKey(0)"""
                            """img=np.zeros((500,500,3), np.uint8)
                            for i in range (0,25):
                                img[timeseries[i][0][0]][timeseries[i][0][1]]=(255,255,255)
                            from PIL import Image
                            image = Image.fromarray(img.astype(np.uint8))
                            Image._show(image)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                            cv2.imshow("", img)
                            cv2.waitKey(0)"""
                        # remove last element (it is an empty list)
                        timeseries.pop(25)
                        # remove (0,0) keypoints i.e. not correctly identified ones
                        fix0Keypoints(timeseries)
                        """print(vel)
                        print(acc)"""
                        # normalize the timeseries
                        normTimeseries = normalizeKeypoints(timeseries)
                        h, lL, rL, rA, lA = computeExtraFeatures(normTimeseries)
                        normTimeseries[25] = h
                        normTimeseries[26] = lL
                        normTimeseries[27] = rL
                        normTimeseries[28] = lA
                        normTimeseries[29] = rA
                        # compute joints acceleration
                        vel, acc = computeVelocitiesAccelerations(timeseries, 30)
                        for i in range(len(vel)):
                            normTimeseries[i+30]=vel[i]
                        for i in range(len(acc)):
                            normTimeseries[i+55]=acc[i]
                        #print(timeseriesFile)
                        #create the file with all extracted features
                        with open(timeseriesFile, 'w') as f:
                            #print(normTimeseries, file=f)
                            print(timeseriesFile.split("\\")[len(timeseriesFile.split("\\"))-1][0:3], file=f)
                            for k, v in normTimeseries.items():
                                print(str(k) + ":" + str(v), file=f)


                    # print(normTimeseries)

                    """for instantTime in range(0, 4):
                        img = np.zeros((500, 500, 3), np.uint8)
                        for i in range(0, 25):
                            img[normTimeseries[i][instantTime][1]][normTimeseries[i][instantTime][0]] = (255, 255, 255)
                        from PIL import Image
                
                        image = Image.fromarray(img.astype(np.uint8))
                        Image._show(image)
                        cv2.waitKey(0)"""

                    """img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    cv2.imshow("", img)"""

                    """for imagePath in imagePaths:
                        datum = op.Datum()
                        imageToProcess = cv2.imread(imagePath)
                        datum.cvInputData = imageToProcess
                        opWrapper.emplaceAndPop([datum])
                        cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
                        cv2.waitKey(0)"""
                    """++
                        img = datum.cvInputData
                        for i in datum.poseKeypoints:
                            for j in i:
                                img[int(j[1])][int(j[0])] = (255,255,255)
                
                        cv2.imshow("prova", img)
                        cv2.waitKey(0)
                
                        # Display Image
                        #print("Body keypoints: \n" + str(datum.poseKeypoints))
                        cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
                        cv2.waitKey(0)"""

except Exception as e:
    print(e)
    sys.exit(-1)
