import os
import pickle 
from PIL import Image
import matplotlib.pyplot as plt
dirs = os.listdir('.')
print(dirs)
files = [dir_ for dir_ in dirs if '.pkl' in dir_]
print('Plotting from:', files)
for file_ in files:
    with open(file_, 'rb') as f:
        d = pickle.load(f)
        plt.plot(d)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(file_.split('.')[0])
        plt.savefig(file_.split('.')[0] + '.png')
        plt.close()

