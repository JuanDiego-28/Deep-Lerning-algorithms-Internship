from keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, UpSampling2D, concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.models import Model
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint 
from keras.utils import plot_model



class unet():
    def __init__(self, size, major_kernel_size, image_categories, learning_rate, pretrained_weights=None, pretrained_model=None):
#         self.name= name
        self.inpud_dim= (size,size,3)
        self.learning_rate= learning_rate
        self.major_kernel_size= major_kernel_size
        self.pretrained_weights=pretrained_weights
        #self.pretrained_model= pretrained_model
        self.image_categories= image_categories
        self.build()
        
    def build(self):
        inputs = Input(self.inpud_dim)
        conv1 = Conv2D(filters=32, kernel_size=self.major_kernel_size, strides=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        drop1 = Dropout(0.1)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(drop1) # -> 128x128
        conv2 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        drop2 = Dropout(0.1)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2) # -> 64x64
        conv3 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        drop3 = Dropout(0.1)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3) # -> 32x32
        conv4 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.1)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) # -> 16x16

        conv5 = Conv2D(512, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(512, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.1)(conv5)

        up6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3) # [(None, 25, 25, 512), (None, 24, 24, 512)]
        conv6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(1, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        #conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)
        flatten10 = Flatten()(conv9)
        dense11= Dense(len(self.image_categories), activation = 'softmax')(flatten10)

        self.model = Model(inputs, dense11)

        self.model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy']) # o.2 ok, but slow 0.3? nothing

        self.model.summary()
        
        if(self.pretrained_weights):
            self.model.load_weights(self.pretrained_weights)
            
        return self.model
    
class unet_norm():
    def __init__(self, size, major_kernel_size, image_categories, learning_rate, pretrained_weights=None, pretrained_model=None):
#         self.name= name
        self.inpud_dim= (size,size,3)
        self.learning_rate= learning_rate
        self.major_kernel_size= major_kernel_size
        self.pretrained_weights=pretrained_weights
        #self.pretrained_model= pretrained_model
        self.image_categories= image_categories
        self.build()
        
    def build(self):
        inputs = Input(self.inpud_dim)
        conv1 = Conv2D(filters=32, kernel_size=self.major_kernel_size, strides=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv1 = InstanceNormalization(axis=-1, center=False, scale=False)(conv1)
        drop1 = Dropout(0.1)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(drop1) # -> 128x128
        conv2 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv2 = InstanceNormalization(axis=-1, center=False, scale=False)(conv2)
        drop2 = Dropout(0.1)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2) # -> 64x64
        conv3 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv3 = InstanceNormalization(axis=-1, center=False, scale=False)(conv3)
        drop3 = Dropout(0.1)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3) # -> 32x32
        conv4 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        conv4 = InstanceNormalization(axis=-1, center=False, scale=False)(conv4)
        drop4 = Dropout(0.1)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) # -> 16x16

        conv5 = Conv2D(512, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(512, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        conv5 = InstanceNormalization(axis=-1, center=False, scale=False)(conv5)
        drop5 = Dropout(0.1)(conv5)

        up6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3) # [(None, 25, 25, 512), (None, 24, 24, 512)]
        conv6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        conv6 = InstanceNormalization(axis=-1, center=False, scale=False)(conv6)

        up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        conv7 = InstanceNormalization(axis=-1, center=False, scale=False)(conv7)

        up8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        conv8 = InstanceNormalization(axis=-1, center=False, scale=False)(conv8)

        up9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(1, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = InstanceNormalization(axis=-1, center=False, scale=False)(conv9)
        #conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)
        flatten10 = Flatten()(conv9)
        dense11= Dense(len(self.image_categories), activation = 'softmax')(flatten10)

        self.model = Model(inputs, dense11)

        self.model.compile(optimizer = Adam(learning_rate = self.learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy']) # o.2 ok, but slow 0.3? nothing

        self.model.summary()
        
        if(self.pretrained_weights):
            self.model.load_weights(self.pretrained_weights)
            
        return self.model
    
    
class unet_norm_inverse():
    def __init__(self, size, major_kernel_size, image_categories, learning_rate, pretrained_weights=None, pretrained_model=None):
#         self.name= name
        self.inpud_dim= (size,size,3)
        self.learning_rate= learning_rate
        self.major_kernel_size= major_kernel_size
        self.pretrained_weights=pretrained_weights
        #self.pretrained_model= pretrained_model
        self.image_categories= image_categories
        self.build()
        
    def build(self):
        inputs = Input(self.inpud_dim)
        conv1 = Conv2D(filters=32, kernel_size=self.major_kernel_size, strides=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        conv1 = InstanceNormalization(axis=-1, center=False, scale=False)(conv1)
        drop1 = Dropout(0.1)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(drop1) # -> 128x128
        conv2 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        conv2 = InstanceNormalization(axis=-1, center=False, scale=False)(conv2)
        drop2 = Dropout(0.1)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2) # -> 64x64
        conv3 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        conv3 = InstanceNormalization(axis=-1, center=False, scale=False)(conv3)
        drop3 = Dropout(0.1)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3) # -> 32x32
        conv4 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        conv4 = InstanceNormalization(axis=-1, center=False, scale=False)(conv4)
        drop4 = Dropout(0.1)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) # -> 16x16

        conv5 = Conv2D(512, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(512, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        conv5 = InstanceNormalization(axis=-1, center=False, scale=False)(conv5)
        drop5 = Dropout(0.1)(conv5)

        up6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([up6, pool3], axis = 3) # [(None, 25, 25, 512), (None, 24, 24, 512)]
        conv6 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        conv6 = InstanceNormalization(axis=-1, center=False, scale=False)(conv6)

        up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([up7, pool2], axis = 3)
        conv7 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        conv7 = InstanceNormalization(axis=-1, center=False, scale=False)(conv7)

        up8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([up8, pool1], axis = 3)
        conv8 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        conv8 = InstanceNormalization(axis=-1, center=False, scale=False)(conv8)

        up9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([up9, conv1], axis = 3)
        conv9 = Conv2D(3, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(1, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = InstanceNormalization(axis=-1, center=False, scale=False)(conv9)
        #conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)
        flatten10 = Flatten()(conv9)
        dense11= Dense(len(self.image_categories), activation = 'softmax')(flatten10)

        self.model = Model(inputs, dense11)

        self.model.compile(optimizer = Adam(learning_rate = self.learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy']) # o.2 ok, but slow 0.3? nothing

        self.model.summary()
        
        if(self.pretrained_weights):
            self.model.load_weights(self.pretrained_weights)
            
        return self.model
    
    
class unet2():
    def __init__(self, size, major_kernel_size, image_categories, learning_rate, pretrained_weights=None, pretrained_model= None):
#         self.name= name
        self.inpud_dim= (size,size,3)
        self.learning_rate= learning_rate
        self.major_kernel_size= major_kernel_size
        self.pretrained_weights=pretrained_weights
        #self.pretrained_model= pretrained_model
        self.image_categories= image_categories
        self.build()
        
    def build(self):
        inputs = Input(self.inpud_dim)
        conv1 = Conv2D(filters=32, kernel_size=self.major_kernel_size, strides=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        drop1 = Dropout(0.5)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(drop1) # -> 128x128
        conv2 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        drop2 = Dropout(0.5)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2) # -> 64x64
        conv3 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3) # -> 32x32
        conv4 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) # -> 16x16

        conv5 = Conv2D(512, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(512, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        pool5 = MaxPooling2D(pool_size=(2, 2))(drop5) # -> 8x8
        conv6 = Conv2D(512, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5)
        conv6 = Conv2D(512, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        drop6 = Dropout(0.5)(conv6)
        up6_1 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop6))
        merge6_1 = concatenate([drop5,up6_1], axis = 3) # [(None, 25, 25, 512), (None, 24, 24, 512)]
        conv6_1 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6_1)
        conv6_1 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6_1)
        drop6 = Dropout(0.5)(conv6_1)
        
        
        up6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop6))
        merge6 = concatenate([drop4,up6], axis = 3) # [(None, 25, 25, 512), (None, 24, 24, 512)]
        conv6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(1, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        #conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)
        flatten10 = Flatten()(conv9)
        dense11= Dense(len(self.image_categories), activation = 'softmax')(flatten10)

        self.model = Model(inputs, dense11)

        self.model.compile(optimizer = Adam(learning_rate = self.learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy']) # o.2 ok, but slow 0.3? nothing

        self.model.summary()
        
        if(self.pretrained_weights):
            self.model.load_weights(self.pretrained_weights)
            
        return self.model
    
    
class mask_unet():
    def __init__(self, size, major_kernel_size, learning_rate, pretrained_weights=None, pretrained_model=None):
#         self.name= name
        self.inpud_dim= (size,size,3)
        self.learning_rate= learning_rate
        self.major_kernel_size= major_kernel_size
        self.pretrained_weights=pretrained_weights
        #self.pretrained_model= pretrained_model
        self.build()
        
    def build(self):
        inputs = Input(self.inpud_dim)
        conv1 = Conv2D(filters=32, kernel_size=self.major_kernel_size, strides=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        drop1 = Dropout(0.1)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(drop1) # -> 128x128
        conv2 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        drop2 = Dropout(0.1)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2) # -> 64x64
        conv3 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        drop3 = Dropout(0.1)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3) # -> 32x32
        conv4 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.1)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) # -> 16x16

        conv5 = Conv2D(512, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(512, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.1)(conv5)

        up6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4,up6], axis = 3) # [(None, 25, 25, 512), (None, 24, 24, 512)]
        conv6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(256, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(128, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(64, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = concatenate([conv1,up9], axis = 3)
        conv9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(32, self.major_kernel_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        #conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)
        output = Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid', kernel_initializer = 'he_normal', name='decoder_out')(conv9)

        self.model = Model(inputs, output)

        self.model.compile(optimizer = Adam(learning_rate = 0.001, beta_1=0.90, beta_2=0.99), loss="binary_crossentropy", metrics = ['accuracy'])

        self.model.summary()
        
        if(self.pretrained_weights):
            self.model.load_weights(self.pretrained_weights)
            
        return self.model
    

