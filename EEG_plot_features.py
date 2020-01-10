import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os 

PATH = os.getcwd()

mw = np.load(PATH +'/data/EEG_com_crise.npy').T
mn = np.load(PATH +'/data/EEG_sem_crise.npy').T

classes = np.hstack((np.repeat("Yes",len(mw)), np.repeat("No", len(mn))))
y = np.hstack((
    np.zeros(len(mw)).astype(int),
    np.ones(len(mn)).astype(int)
    ))

col = ['Kurtosis','Skewness','Variance','Energy']
data = np.vstack((mw,mn))

D = pd.DataFrame(data=data,columns=col)

D['Seizures'] = classes

print(D.head())

# Boxplot -----------------------------------------------
# fig = plt.figure()
# gs = GridSpec(ncols=2,nrows=2,figure=fig)

# ax1 = fig.add_subplot(gs[0,0])
# sns.boxplot(x='Seizures', y='Kurtosis',
#             hue="Seizures", data=D, showfliers=False,ax=ax1)

# ax2 = fig.add_subplot(gs[0,1])
# sns.boxplot(x='Seizures', y='Skewness',
#             hue="Seizures", data=D, showfliers=False,ax=ax2)

# ax3 = fig.add_subplot(gs[1,0])
# sns.boxplot(x='Seizures', y='Variance',
#             hue="Seizures", data=D, showfliers=False,ax=ax3)

# ax4 = fig.add_subplot(gs[1,1])
# sns.boxplot(x='Seizures', y='Energy',
#             hue="Seizures", data=D, showfliers=False,ax=ax4)



# Pair plot -------------------------------------------
#fig2 = plt.figure()
#sns.pairplot(D,hue='Seizures')

# 3D plot --------------------------------------------

D1 = D[D['Seizures'] == 'Yes']
D2 = D[D['Seizures'] == 'No']

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(D1['Variance'],D1['Energy'],D1['Kurtosis'],label='With seizures')
# ax.scatter(D2['Variance'],D2['Energy'],D2['Kurtosis'],label='Without seizures')
# ax.set_xlabel(col[2])
# ax.set_ylabel(col[3])
# ax.set_zlabel(col[0])

# plt.legend()


# Scatter plot
fig = plt.figure()

plt.subplot(2,2,1)
plt.plot(D1['Variance'].to_numpy(),'.')
plt.plot(D2['Variance'].to_numpy(),'.')

plt.subplot(2,2,2)
plt.plot(D1['Energy'].to_numpy(),'.')
plt.plot(D2['Energy'].to_numpy(),'.')

plt.subplot(2,2,3)
plt.plot(D1['Kurtosis'].to_numpy(),'.')
plt.plot(D2['Kurtosis'].to_numpy(),'.')

plt.subplot(2,2,4)
plt.plot(D1['Skewness'].to_numpy(),'.')
plt.plot(D2['Skewness'].to_numpy(),'.')

plt.show()