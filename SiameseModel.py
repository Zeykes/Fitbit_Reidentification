from tensorflow.keras import models , optimizers , losses ,activations , callbacks
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from PIL import Image
import tensorflow as tf
import time
import os
import numpy as np


class Recognizer (object) :

	def __init__( self ):

		input_shape = ( 17280, 8 ) # 1-day of 5-sec frequency values
		# increase kernel_sizes
		kernel_size_1 = ( 64 )
		kernel_size_2 = ( 64 )
		pool_size_1 = ( 3 )
		pool_size_2 = ( 2 )
		strides = 1

		seq_conv_model = [

			Conv1D( 32, kernel_size=kernel_size_1 , strides=strides , activation=self.leaky_relu ),
			Conv1D( 32, kernel_size=kernel_size_1, strides=strides, activation=self.leaky_relu),
			#MaxPooling1D(pool_size=pool_size_1, strides=strides ),

			Conv1D( 64, kernel_size=kernel_size_2 , strides=strides , activation=self.leaky_relu ),
			Conv1D( 64, kernel_size=kernel_size_2 , strides=strides , activation=self.leaky_relu ),
			#MaxPooling1D(pool_size=pool_size_2 , strides=strides),

			#Flatten(),

			Dense( 64 , activation=activations.sigmoid )

		]

		seq_model = tf.keras.Sequential( seq_conv_model )

		input_x1 = Input( shape=input_shape )
		input_x2 = Input( shape=input_shape )

		output_x1 = seq_model( input_x1 )
		output_x2 = seq_model( input_x2 )
		seq_model.summary()

		distance_euclid = Lambda( lambda tensors : K.abs( tensors[0] - tensors[1] ))( [output_x1 , output_x2] )
		outputs = Dense( 1 , activation=activations.sigmoid) ( distance_euclid )
		self.__model = models.Model( [ input_x1 , input_x2 ] , outputs )

		self.__model.compile( loss=losses.binary_crossentropy , optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])

	def leaky_relu(self, x):
		return tf.nn.leaky_relu(x, alpha=0.01)

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
