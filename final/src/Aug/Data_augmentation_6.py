print('Program start')

### include library ###
import os
from time import time
from PIL import Image # pip install pillow
from scipy.misc import imsave
import numpy as np
import pandas as pd
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator

### get input index ###
Indexxx = 6
Rotate = [1,11]
Shift = [1,6]
Shear = [1,5]
Zoom = [1,5]

### related paths ###
ListPath = './trainnn.csv'
PictureDirectory = '../../images/'
SavingDirectory = '../../images/' # output picture directory

### known parameters ###
Data = {'width':1024,'channel':1}
Label = {'category':14}

### hyper parameters ###
AugmentationPara = {'index':[Indexxx],'batch':256}
FirstLook = False

### data generator ###
# keras DataGenerator method (keras.utils.Sequence + keras.models.fit_generator)
# <keras document> https://keras.io/zh/utils/#sequence
# <source code copyright> https://blog.csdn.net/m0_37477175/article/details/79716312
class DataGenerator(Sequence): 
    def __init__(self, list_IDs, labels, batch_size=32, dim=1024, n_channels=1, n_classes=14, shuffle=False): 
        self.dim = dim 
        self.batch_size = batch_size 
        self.labels = labels 
        self.list_IDs = list_IDs 
        self.n_channels = n_channels 
        self.n_classes = n_classes 
        self.shuffle = shuffle 
        self.on_epoch_end() 
        # X : (n_samples, dim, dim, n_channels) 
        # Initialization 
        self.X = np.empty((batch_size, dim, dim, n_channels)) 
        self.y = np.empty((batch_size, n_classes)) 
    def __len__(self): # Denotes the number of batches per epoch
        return int(np.floor(len(self.list_IDs) / self.batch_size)) 
    def __getitem__(self, index): # Generate one batch of data 
        # Generate indexes of the batch 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] 
        # Find list of IDs 
        list_IDs_temp = [self.list_IDs[k] for k in indexes] 
        # Generate data 
        self.X, self.y = self.__data_generation(list_IDs_temp) 
        return self.X, self.y 
    def on_epoch_end(self): #Updates indexes after each epoch 
        self.indexes = np.arange(len(self.list_IDs)) 
        if self.shuffle == True: 
            np.random.shuffle(self.indexes) 
    def __data_generation(self, list_IDs_temp): #Generates data containing batch_size samples 
        # Generate data 
        for i, ID in enumerate(list_IDs_temp): 
            # Store sample 
            x = np.array(Image.open(PictureDirectory+ID))
            if x.size != 1048576:
                x = x[:,:,0]
            self.X[i] = x.reshape(self.dim, self.dim, self.n_channels)
            # Store class 
            self.y[i] = self.labels[ID]
        return self.X, self.y

### data preprocessing ###
List = pd.read_csv(ListPath)
List.set_index('Image Index', inplace=True)
List = List['Labels']
ListID = np.array(List.index)
MultiHotLabel = []
for labellist in List:
    MultiHotLabel.append(np.array(labellist.split(' ')).astype('int'))
MultiHotLabel = pd.Series(MultiHotLabel,index=List.index)
training_generator = DataGenerator(ListID,MultiHotLabel,
                                   AugmentationPara['batch'],Data['width'],Data['channel'],Label['category'],shuffle=False)

### image generator ###
imgg = ImageDataGenerator(
        #featurewise_center=False, 
        #samplewise_center=False,
        #featurewise_std_normalization=False, 
        #samplewise_std_normalization=False, 
        #zca_whitening=False, 
        #zca_epsilon=1e-06, 
        rotation_range=0.01*np.random.randint(Rotate[0],Rotate[1]),
        width_shift_range=0.01*np.random.randint(Shift[0],Shift[1]),
        height_shift_range=0.01*np.random.randint(Shift[0],Shift[1]),
        #brightness_range=None, 
        shear_range=0.01*np.random.randint(Shear[0],Shear[1]),
        zoom_range=0.01*np.random.randint(Zoom[0],Zoom[1]),
        #channel_shift_range=0.0, 
        #fill_mode='nearest', 
        #cval=0.0, 
        horizontal_flip=True, 
        vertical_flip=False, 
        #rescale=None, 
        #preprocessing_function=None, 
        #data_format=None, 
        #validation_split=0.0
)

### image augmentation & saving ###
try:
    os.makedirs(SavingDirectory)
except:
    None
NewID = []
NewLabel = []
StartTime = time()
for c in AugmentationPara['index']:
    for i in range(int(len(ListID)/AugmentationPara['batch'])):
        real_pic = training_generator.__getitem__(i)
        xx = real_pic[0]
        yy = real_pic[1]
        aug_pic = next(imgg.flow(xx,yy,batch_size=xx.shape[0],shuffle=False))
        x = aug_pic[0]
        y = aug_pic[1]
        for j,p in enumerate(x):
            filename = 'aug_{}_'.format(c)+List.index[j+i*AugmentationPara['batch']]
            imsave(SavingDirectory+filename,p.reshape(Data['width'],Data['width']))
            NewID.append(filename)
            NewLabel.append(' '.join([str(int(s)) for s in y[j]]))
        print('index:',c,
              '  pictures done(this copy):',1+j+i*AugmentationPara['batch'],
              '  total time elapsed:','{0:.2f}sec'.format(time()-StartTime))
        if FirstLook == True:
            break
                                                                                  
print('Program end')
