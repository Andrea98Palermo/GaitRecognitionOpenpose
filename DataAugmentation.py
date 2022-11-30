import os
import cv2

class DataAugmentation:

    def flipAugmentation(self, dbRoot="D:\\gaitdb"):
        newDbRoot = dbRoot+"Flip"
        try:
            os.mkdir(newDbRoot)
        except OSError:
            None
        for username in os.listdir(dbRoot):
            for user in username:
                if user != "video" and user != "member_photo":
                    userRoot = dbRoot + os.sep + username
                    newUserRoot = newDbRoot + os.sep + username
                    try:
                        os.mkdir(newUserRoot)
                    except OSError:
                        None
                    for walk in os.listdir(userRoot):
                        walkRoot = userRoot + os.sep + walk
                        newWalkRoot = newUserRoot + os.sep + walk
                        try:
                            os.mkdir(newWalkRoot)
                        except OSError:
                            None
                        for image in os.listdir(walkRoot):
                            img = cv2.imread(walkRoot+os.sep+image)
                            imgName=image[:-4]+"flip"+image[-4:]
                            flip = cv2.flip(img, 1)
                            cv2.imwrite(newWalkRoot + os.sep + imgName, flip)

if __name__ == '__main__':
    dataAug=DataAugmentation()
    dataAug.flipAugmentation()