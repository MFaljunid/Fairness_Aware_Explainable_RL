import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import glob

pkl_files = glob.glob('results/BPR/*.pkl')
with open(sorted(pkl_files)[-1], 'rb') as f:
    bpr = pickle.load(f)

print("BPR model attributes:")
for attr in dir(bpr):
    if not attr.startswith('__'):
        print(f"  {attr}")