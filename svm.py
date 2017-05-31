import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
import os

# path to folder containing all the images
all_folders = os.listdir("files")

train_files = []
train_labels = []


def resize(array, new_size, new_value=0):
    """Resize to biggest or lesser size."""
    element_size = len(array) #Quantity of new elements equals to quantity of first element
    if new_size > len(array):
        new_size = new_size - 1
        while len(array)<=new_size:
            n = tuple(new_value for i in range(element_size))
            array.append(n)
    else:
        array = array[:new_size]
    return array

def normalize_data(data):
    temp=[]
    for r in data['data']:
        for i in r:
            #print(r[i]['rpm'])
            # print(r[i]['speed'])
            if (int(i['rpm']) > 3000 or (int(i['speed']) > int(i['maximum_limition_of_speed']) and int(i['maximum_limition_of_speed']) != -1)):
                temp.append(int(i['rpm']) * 0.001 + (int(i['speed']) - int(i['maximum_limition_of_speed'])))  # 1
            else:
                temp.append(0)  # 0

    temp = resize(temp, 200)
    return temp;


for folder in all_folders:
    files = os.listdir("files/" + folder)
    folder_files = []  # only the images of one folder
    folder_labels = []
    for file in files:
        with open("files/" + folder + "/" + file) as json_data:
            data = json.load(json_data)
        #print(normalize_data(data))
        folder_files.append(normalize_data(data))
        if(folder == '0'):
            folder_labels.append(0)
        if(folder == '1'):
            folder_labels.append(1)

    train_files.extend(folder_files[:len(folder_files)])
    train_labels.extend(folder_labels[:len(folder_files)])

print(train_files)
print(train_labels)


#prediction starts here
with open('predict/test.json') as json_data:
    pre_data = json.load(json_data)



clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(train_files,train_labels);

print(clf.predict(normalize_data(pre_data)))