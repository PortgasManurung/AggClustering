import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

#ambil dataset
dataset = pd.read_csv("dataset.csv")
dataset.head()


#Menentukan variabel yang akan di klusterkan 
dataset_x = dataset.iloc[:, 1:3]
dataset_x.head()
#print(dataset_x.head())


#Mengubah Variabel Data Frame Menjadi Array 
x_array =  np.array(dataset_x)
#print(x_array)

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x_array)
x_scaled

#Menentukan dan mengkonfigurasi fungsi Agg
agg = AgglomerativeClustering(n_clusters = 3, affinity='euclidean', linkage='ward')


#Menentukan kluster dari data
agg.fit(x_scaled)

#Menampilkan Hasil Kluster 
#print(agg.labels_)

#Menambahkan Kolom "kluster" Dalam Data Frame Driver
dataset["kluster"] = agg.labels_

output = plt.scatter(x_scaled[:,0], x_scaled[:,1], s = 100, c = dataset.kluster, marker = "o", alpha = 1, )
plt.title("AgglomerativeClustering")
plt.colorbar (output)
plt.show()