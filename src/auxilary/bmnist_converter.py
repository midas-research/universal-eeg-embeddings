import pandas as pd
import numpy as np

"""
Code to extract data from BMnist into the format
samples x channel x observations

perform trimming for samples to the minimum length
"""

df = pd.read_csv('MU.txt', sep='\t', header=None)
print(df.head())

df.columns = ['id', 'event', 'headset', 'channel', 'truth', 'len', 'raw']

min_len = np.min(df['len'])

arr = []
truths = []
for event, rest in df.groupby('event'):
    # Rest is a one of the grouped things
    # event is the event id
    mini_arr = []
    truth = None    # use any event ID in order to work
    for i, row in rest.iterrows():
        truth = row['truth']
        mini_arr.append([int(x) for x in row['raw'].split(',')[:min_len]])
    truths.append(truth)
    arr.append(mini_arr)
print(len(arr))
print(len(truths))
print(len(np.unique(df['event'])))

X = np.array(arr)
Y = np.array(truths)
print(X.shape)
print(Y.shape)

# Use numpy's inbuilt pickling to save the data
with open('./bmnist_x.npy', 'wb') as f:
    np.save(f, X)

with open('./bmnist_y.npy', 'wb') as f:
    np.save(f, Y)
