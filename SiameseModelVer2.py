from tensorflow.keras import models , optimizers , losses ,activations , callbacks
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import time
import os
import numpy as np


class Recognizer (object) :

    def __init__( self ):

        #input_shape = ( 10080, 1 ) # Day: 1440, Week: 10080
        input_shape = ( 17280, 8 ) # 1-day of 5-sec frequency values
        # ( 7, 1440 )
        # ( 1440, 7 )
        # increase kernel_sizes
        kernel_size_1 = ( 32 )
        kernel_size_2 = ( 32 )
        pool_size_1 = ( 2 )
        pool_size_2 = ( 2 )
        strides = 1

		# Define the tensors for the two input images
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        # Convolutional Neural Network
        model = tf.keras.Sequential()
        model.add(Conv1D(32, (10), activation='relu', input_shape=input_shape,
                        kernel_initializer=self.initialize_weights, kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling1D())
        model.add(Conv1D(64, (7), activation='relu',
                        kernel_initializer=self.initialize_weights,
                        bias_initializer=self.initialize_bias, kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling1D())
        model.add(Conv1D(64, (4), activation='relu', kernel_initializer=self.initialize_weights,
                        bias_initializer=self.initialize_bias, kernel_regularizer=l2(2e-4)))
        model.add(MaxPooling1D())
        model.add(Conv1D(128, (4), activation='relu', kernel_initializer=self.initialize_weights,
                        bias_initializer=self.initialize_bias, kernel_regularizer=l2(2e-4)))
        model.add(Flatten())
        model.add(Dense(2048, activation='sigmoid',
                        kernel_regularizer=l2(1e-3),
                        kernel_initializer=self.initialize_weights,bias_initializer=self.initialize_bias))

        # Generate the encodings (feature vectors) for the two images
        encoded_l = model(left_input)
        encoded_r = model(right_input)

        # Add a customized layer to compute the absolute difference between the encodings
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])

        # Add a dense layer with a sigmoid unit to generate the similarity score
        prediction = Dense(1,activation='sigmoid',bias_initializer=self.initialize_bias)(L1_distance)

        # Connect the inputs with the outputs
        self.__model = models.Model(inputs=[left_input,right_input],outputs=prediction)

        self.__model.compile( loss="binary_crossentropy" , optimizer=optimizers.Adam(lr=0.00006), metrics=['accuracy'])

    def leaky_relu(self, x):
        return tf.nn.leaky_relu(x, alpha=0.01)
    
    def initialize_weights(self, shape, dtype=None):
        """
            The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
            suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
        """
        return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)
    
    def initialize_bias(self, shape, dtype=None):
        """
            The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
            suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
        """
        return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

    def fit(self, X, Y ,  hyperparameters  ):
        initial_time = time.time()
        history = self.__model.fit( X  , Y ,
						 batch_size=hyperparameters[ 'batch_size' ] ,
						 epochs=hyperparameters[ 'epochs' ] ,
						 callbacks=hyperparameters[ 'callbacks'],
						 validation_data=hyperparameters[ 'val_data' ]
						 )
        final_time = time.time()
        eta = ( final_time - initial_time )
        time_unit = 'seconds'
        if eta >= 60 :
            eta = eta / 60
            time_unit = 'minutes'
        self.__model.summary( )
        print( 'Elapsed time acquired for {} epoch(s) -> {} {}'.format( hyperparameters[ 'epochs' ] , eta , time_unit ) )
        return history


    def evaluate(self , test_X , test_Y  ) :
        return self.__model.evaluate(test_X, test_Y)


    def predict(self, X  ):
        predictions = self.__model.predict( X  )
        return predictions


    def summary(self):
        self.__model.summary()


    def save_model(self , file_path ):
        self.__model.save(file_path )


    def load_model(self , file_path ):
        self.__model = models.load_model(file_path)
