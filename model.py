import scipy
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
import sklearn.model_selection as sk
import pickle

from tensorflow.keras import layers, models


rank_data = pickle.load(open('rank_data.pickle', 'rb'))
num_sensors=16

sample_rate=512

master_avg_diffs=0
num_patients=16


for patient in range(num_patients):
    condition1=rank_data[patient][2]
    condition2=rank_data[patient][3]*2
    conditions=condition1+condition2
    cond1_ind=np.where(conditions==1)[0]
    cond2_ind=np.where(conditions==2)[0]
    diffs=(cond2_ind-cond1_ind) if cond2_ind[0]>cond1_ind[0] else (cond1_ind-cond2_ind)
    avg_diffs=sum(diffs)/len(diffs)
    master_avg_diffs+=avg_diffs


master_avg_diffs/=num_patients
#divide by two since we only want to get onset of change
master_avg_diffs=int(master_avg_diffs/2)

data=[]
labels=[]

for patient in range(8):
    condition1=rank_data[patient][2]
    condition2=rank_data[patient][3]*2
    conditions=condition1+condition2
    cond1_ind=np.where(conditions==1)[0]
    cond2_ind=np.where(conditions==2)[0]
    start_at_cond1=True if cond1_ind[0]<cond2_ind[0] else False

    slice_data=[[] for _ in range(len(cond1_ind)*2)]
    for i in range(num_sensors):
        for j in range(len(cond1_ind)):
            start = (cond1_ind[j] if start_at_cond1 else cond2_ind[j])
            end = start+master_avg_diffs
            Zxx,f,t,im=plt.specgram(rank_data[patient][1][i][start:end],Fs=sample_rate)
            slice_data[2*j].append(Zxx)
            
            start = end
            end = start+master_avg_diffs
            Zxx,f,t,im=plt.specgram(rank_data[patient][1][i][start:end],Fs=sample_rate)
            slice_data[2*j+1].append(Zxx)



    for i in range(len(slice_data)):
        img=np.hstack(slice_data[i])
        data.append(img)
        if start_at_cond1:  
            if (i//2)%2==0:
                labels.append(1)
            else:
                labels.append(2)
        else:
            if (i//2)%2==0:
                labels.append(2)
            else:
                labels.append(1)
        

X_train, X_test, y_train, y_test = sk.train_test_split(data,labels,test_size=0.2, random_state = 42)


X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)



input_shape=X_train[0].shape+(1,)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print("Accuracy:",test_acc*100,"%")