import numpy as np
import pickle

f1_dir = 'accuracy.txt'
f2_dir = 'loss.txt'

f1 = open(f1_dir, 'rb')
accuracy = pickle.load(f1)
f1.close()
f2 = open(f2_dir, 'rb')
Loss = pickle.load(f2)
f2.close()

