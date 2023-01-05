import cv2
import face_recognition
import os
import pickle

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://faceattendencerealtime-50ecf-default-rtdb.firebaseio.com/",
    "storageBucket": "faceattendencerealtime-50ecf.appspot.com"
})

# Importing the students images
folderPath = 'Images'
pathList = os.listdir(folderPath)
# print(pathList)
imgList = []
studentIds = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

    # print(path)
    # print(os.path.splitext(path)[0])

print(studentIds)


def findEncodeings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


print("Encoding started  ... ")
encodeListKnown = findEncodeings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print(encodeListKnown)
print("Encoding ended")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")
