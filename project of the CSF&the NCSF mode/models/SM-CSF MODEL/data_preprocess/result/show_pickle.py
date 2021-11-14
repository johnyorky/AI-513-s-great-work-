import pickle
import sys
file_name=sys.argv[1]
with open(file_name,"rb")as f:
    data=pickle.load(f)
print(data)

