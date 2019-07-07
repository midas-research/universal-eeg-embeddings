import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

"""
Example of how to read data supplied by ThoughtViz and create
small plots to visualise it
"""

char_d = pkl.load(open('/media/data_dump_1/group_2/data/thoughtviz/char/data.pkl', 'rb'), encoding='bytes')
print('Character Data')
print(char_d[b'x_train'].shape, char_d[b'y_train'].shape)
print(char_d[b'x_test'].shape, char_d[b'y_test'].shape)

digit_d = pkl.load(open('/media/data_dump_1/group_2/data/thoughtviz/digit/data.pkl', 'rb'), encoding='bytes')
print('Digit Data')
print(digit_d[b'x_train'].shape, digit_d[b'y_train'].shape)
print(digit_d[b'x_test'].shape, digit_d[b'y_test'].shape)

img_d = pkl.load(open('/media/data_dump_1/group_2/data/thoughtviz/image/data.pkl', 'rb'), encoding='bytes')
print('Image Data')
print(img_d[b'x_train'].shape, img_d[b'y_train'].shape)
print(img_d[b'x_test'].shape, img_d[b'y_test'].shape)

print(type(char_d))
print(np.average(char_d[b'x_train'][0,:], axis=1))
plt.plot(np.average(char_d[b'x_train'][0,:], axis=0), label='Character')
plt.plot(np.average(digit_d[b'x_train'][0,:], axis=0), label='Digit')
plt.plot(np.average(img_d[b'x_train'][0,:], axis=0), label='Image')
plt.legend()
plt.show()
