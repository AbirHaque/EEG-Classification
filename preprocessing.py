import scipy
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
import pickle
from mpi4py import MPI
import time



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start = time.time()

num_patients=16
offset=4
num_in_chunk=num_patients//size
low=(rank*num_in_chunk)+offset
high=((rank+1)*num_in_chunk if rank!=size-1 else num_patients)+offset
rank_data=[]


for num in range(low,high):
    num_str=str(num)
    if num < 10:
        num_str="0"+num_str
    mat_fname = 'subject_'+num_str+'.mat'
    mat_data = scipy.io.loadmat(mat_fname)['SIGNAL'].T
    condition1=mat_data[17]
    condition2=mat_data[18]
    timesteps=mat_data[0]
    num_sensors=None

    observed_signal_data=None

    num_sensors=16
    observed_signal_data=mat_data[1:num_sensors+1]
    observed_signal_data/=observed_signal_data.std(axis=0)
    num_samples=len(observed_signal_data[0])
    sample_rate=512

    ica = FastICA(n_components=num_sensors,max_iter=10000, tol=1e-7)
    reconstructed_signal_data = ica.fit_transform(observed_signal_data[:512].T).T
    print(rank,":",num_str)
    
    rank_data.append((observed_signal_data,reconstructed_signal_data,condition1,condition2))



comm.barrier()
rank_data=comm.gather(rank_data,root=0)
comm.barrier()

end = time.time()


if(rank==0):
    print(end-start)


if rank==0:
    patients=[]
    for i in range(size):
        for j in range(len(rank_data[i])):
            patients.append(rank_data[i][j])    
    pickle.dump(patients, open('rank_data.pickle', 'wb'))


comm.barrier()
