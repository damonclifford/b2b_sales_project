import sys, os

# get current directory 
path = os.getcwd()  
# parent directory 
parent = os.path.dirname(path) 


# Adds the root folder to the python path
sys.path.insert(1, os.path.join(sys.path[0], 'utils/'))
sys.path.insert(1, os.path.join(sys.path[0], 'backup/'))


# Adds the support folder to the python path
sys.path.insert(1, os.path.join(parent, 'models_pkl/'))
sys.path.insert(1, os.path.join(parent, 'support/'))



