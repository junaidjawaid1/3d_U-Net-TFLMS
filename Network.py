import Network_elements_instance 
import Network_elements_instance 
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv3D, InputLayer
import Data_Transform
import matplotlib.pyplot as plt

tf.config.experimental.set_lms_enabled(True)

block=Network_elements_instance.blocks()

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def cartesian_product(*arrays):
    ndim = len(arrays)
    return np.stack(np.meshgrid(*arrays), axis=-1).reshape(-1, ndim)


def Unet_3d(x_in):

    Multi_Elab_1=block.MultiScaleElaboration(x_in)
    Multi_Elab_1_1=block.MultiScaleElaboration(Multi_Elab_1)    #To be used for concatenation

    Reduction_1=block.Reduction(Multi_Elab_1_1)

    Multi_Elab_2=block.MultiScaleElaboration(Reduction_1)
    Multi_Elab_2_1=block.MultiScaleElaboration(Multi_Elab_2)   #To be used for concatenation

    Reduction_2=block.Reduction(Multi_Elab_2_1)

    Multi_Elab_3=block.MultiScaleElaboration(Reduction_2)
    Multi_Elab_3_1=block.MultiScaleElaboration(Multi_Elab_3)  #To be used for concatenation

    Reduction_3=block.Reduction(Multi_Elab_3_1)

    Multi_Elab_4=block.MultiScaleElaboration(Reduction_3)
    Multi_Elab_4_1=block.MultiScaleElaboration(Multi_Elab_4)

    Expansion_1=block.expansion(Multi_Elab_4_1)

    Concat_1=Concatenate(axis=-1)([Multi_Elab_3_1,Expansion_1])

    Multi_scale_elaboration_5=block.MultiScaleElaboration(Concat_1)
    Multi_scale_elaboration_5_1=block.MultiScaleElaboration(Multi_scale_elaboration_5)

                               
    Expansion_2=block.expansion(Multi_scale_elaboration_5_1)

    Concat=Concatenate(axis=-1)([Multi_Elab_2_1,Expansion_2])

    multi_scale_elaboration_6=block.MultiScaleElaboration(Concat)
    multi_scale_elaboration_6_1=block.MultiScaleElaboration(multi_scale_elaboration_6)

    Expansion_3=block.expansion(multi_scale_elaboration_6_1)

    Concat=Concatenate(axis=-1)([Multi_Elab_1_1,Expansion_3])

    multi_scale_elaboration_7=block.MultiScaleElaboration(Concat)
    multi_scale_elaboration_7_1=block.MultiScaleElaboration(multi_scale_elaboration_7)

    Conv=Conv3D (filters=1, 
                       kernel_size=[1,1,1], 
                       strides=[1,1,1],
                       data_format="channels_last",
                       use_bias= True,
                       dtype=np.float32)(multi_scale_elaboration_7_1)
    
    return Conv

window_width = 16

input = tf.keras.Input(shape=(512,window_width,512,2))

Unet_output=Unet_3d(input)

print(np.shape(Unet_output))

model = tf.keras.Model(inputs=input, outputs=Unet_output)

# Data pipeline

transform = Data_Transform.transformation()

class DataGenerator(tf.keras.utils.Sequence):
    'Generate data for keras'

    def __init__(self, path, batch_size, triple_list, dim , n_channel = 2,
                 shuffle = True):
        
        self.dim= dim
        self.batch_size= batch_size
        self.triple_list = triple_list
        self.n_channels= n_channel
        self.shuffle= shuffle
        self.path= path
        self.width = dim[1]
        self.on_epoch_end()

    def __len__(self):
        
        elements = np.shape(self.triple_list)
        return int(np.floor(elements[0] / self.batch_size))
    
    def __getitem__(self,idx):

        'Generate a batch of data'
        indexes= self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        index_list_temp= [self.triple_list[k] for k in indexes]

        X_in, Y_in = self.__data_generator(index_list_temp)

        print(X_in.shape, Y_in.shape)

        return X_in, Y_in

    def on_epoch_end(self):
        self.indexes= np.arange(len(self.triple_list))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generator(self, triple_list):

        X = np.empty((self.batch_size, *self.dim, self.n_channels))   # 7 corresponds to data augmentation
        Y = np.empty((self.batch_size, *self.dim, 1)) 

        for i, ID in enumerate(triple_list):

            CT_scan = np.load(self.path+'/CT_scans'+'/CT_'+ID[0]+'.npy').astype(np.float32)
            Dose_5K = np.load(self.path+'/Dose_5K'+'/Dose5K_'+ID[0]+'.npy').astype(np.float32)
            Dose_1M = np.load(self.path+'/Dose_1M'+'/Dose1M_'+ID[0]+'.npy').astype(np.float32)

            #Normalization

            CT_scan_norm = (CT_scan + 1000) / 21935
            Dose_5K_norm = (Dose_5K - np.min(Dose_5K)) / (np.max(Dose_5K) - np.min(Dose_5K))
            Dose_1M_norm = (Dose_1M - np.min(Dose_1M)) / (np.max(Dose_1M) - np.min(Dose_1M))

            CT_scan_slice = CT_scan_norm[:, int(ID[1]):int(ID[1])+self.width, :]
            Dose_5K_slice = Dose_5K_norm[:, int(ID[1]):int(ID[1])+self.width, :]
            Dose_1M_slice = Dose_1M_norm[:, int(ID[1]):int(ID[1])+self.width, :]
            
            if ID[2] == '1':
                CT_scan_slice = transform.flip(CT_scan_slice, 0)
                Dose_5K_slice = transform.flip(Dose_5K_slice, 0)
                Dose_1M_slice = transform.flip(Dose_1M_slice, 0)

            elif ID[2] == '2':
                CT_scan_slice = transform.flip(CT_scan_slice, 1)
                Dose_5K_slice = transform.flip(Dose_5K_slice, 1)
                Dose_1M_slice = transform.flip(Dose_1M_slice, 1)

            elif ID[2] == '3':
                CT_scan_slice = transform.flip(CT_scan_slice, 2)
                Dose_5K_slice = transform.flip(Dose_5K_slice, 2)
                Dose_1M_slice = transform.flip(Dose_1M_slice, 2)

            elif ID[2] == '4':
                CT_scan_slice = transform.rotation(CT_scan_slice, 1, (0, 2))
                Dose_5K_slice = transform.rotation(Dose_5K_slice, 1, (0, 2))
                Dose_1M_slice = transform.rotation(Dose_1M_slice, 1, (0, 2))

            elif ID[2] == '5':
                CT_scan_slice = transform.rotation(CT_scan_slice, 2, (0, 1))
                Dose_5K_slice = transform.rotation(Dose_5K_slice, 2, (0, 1))
                Dose_1M_slice = transform.rotation(Dose_1M_slice, 2, (0, 1))

            elif ID[2] == '6':
                CT_scan_slice = transform.rotation(CT_scan_slice, 2, (1, 2))
                Dose_5K_slice = transform.rotation(Dose_5K_slice, 2, (1, 2))
                Dose_1M_slice = transform.rotation(Dose_1M_slice, 2, (1, 2))


            X[i,]= np.stack((CT_scan_slice, Dose_5K_slice), axis=-1)
            Y[i,]= Dose_1M_slice.reshape((*self.dim, 1))

        
        return X, Y

train_path = '/NFSHOME/mspezialetti/sharedFolder/U-net_dataset/training'
validation_path = '/NFSHOME/mspezialetti/sharedFolder/U-net_dataset/validation'

train_batch_size = 2

training_examples = 785
index_list_train = [f'{num:03}' for num in range (1, training_examples+1)]
index_list_train = np.array(index_list_train)

slices_list = np.arange(0, 64 - window_width + 1)

transformation_list = np.arange(0, 8)

resulting_list = cartesian_product(index_list_train, slices_list, transformation_list)

prams_train = {'triple_list': resulting_list,
                'dim': (512, window_width, 512),
                'n_channel':2,
                'shuffle':True}

training_generator = DataGenerator(train_path, train_batch_size,**prams_train)

validation_examples = 75
index_list_validation = [f'{num:03}' for num in range (1, validation_examples+1)]
index_list_validation = np.array(index_list_validation)

resulting_list = cartesian_product(index_list_validation, slices_list, transformation_list)

prams_validation = {'triple_list': resulting_list,
                'dim': (512, window_width,512),
                'n_channel':2,
                'shuffle':True}

validation_generator = DataGenerator(validation_path, batch_size = 2,**prams_validation)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None)])    #tf.keras.metrics.Accuracy - does not work

print(model.summary())

history = model.fit(training_generator,
#          batch_size = 2,
          epochs = 10,
          #verbose = 'auto',
          callbacks= [tf.keras.callbacks.CSVLogger('/NFSHOME/mspezialetti/sharedFolder/3D_Unet/training_log5csv', separator = ','),  
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                           patience=2, min_lr=0.0000001, verbose=1)],
                      #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],
          validation_data = validation_generator,
#          validation_batch_size = 2,
          shuffle = True,
          steps_per_epoch = 500,
          validation_steps=200,
          initial_epoch = 0,
          workers = 30,
          verbose=1) # tensorboard can also be included in callbacks


plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model root mean squared error')
plt.ylabel('root mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("/NFSHOME/mspezialetti/sharedFolder/3D_Unet/RMSE_aug.jpg")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("/NFSHOME/mspezialetti/sharedFolder/3D_Unet/MSE_aug.jpg")


testing_path = '/NFSHOME/mspezialetti/sharedFolder/U-net_dataset/testing'
testing_examples = 89
index_list_test = [f'{num:03}' for num in range (1, testing_examples+1)]
index_list_test = np.array(index_list_test)

transformation_list = np.array([0])

resulting_list = cartesian_product(index_list_test, slices_list, transformation_list)
prams_test = {'triple_list': resulting_list,
                'dim': (512, window_width, 512),
                'n_channel':2,
                'shuffle':True}

test_generator = DataGenerator(testing_path, batch_size=7,**prams_test)

score = model.evaluate(test_generator,
                       workers = 20)
                       # batch_size=7)

print(score)

model.save('/NFSHOME/mspezialetti/sharedFolder/3D_Unet/mymodel_Aug.keras')
