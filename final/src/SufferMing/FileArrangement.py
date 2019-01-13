import os
from shutil import copy2
import numpy as np

modeldir = '../../model/'
filedir = './'
filename = {
        'best': '%s.npz',
        'model': 'model_%s_r600800e%s.h5',
        'weight': 'model_weights_%s_r600800e%s.h5',
        'out': '%s.csv',
        }
disease_list = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
        'Consolidation',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Pleural_Thickening',
        'Hernia',
        ]



if not os.path.exists(modeldir):
    os.makedirs(modeldir)

for JJ in disease_list:
    bestesttt = np.load(filedir + filename['best'] % JJ)['MM'] +1
    modelname = filename['model'] % (JJ,bestesttt)
    weightname = filename['weight'] % (JJ,bestesttt)
    
    copy2(filedir+modelname,modeldir+modelname)
    copy2(filedir+weightname,modeldir+weightname)

