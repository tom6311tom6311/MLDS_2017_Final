from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

class CapsGANModel(object):
	def __init__(self, image_shape=[60,60,3], noise_dim=100):
		self.image_shape = image_shape
		self.noise_dim = noise_dim
		self.D = None   # discriminator
		self.G = None   # generator
		self.AM = None  # adversarial model
		self.DM = None  # discriminator model

	def discriminator(self):
		if self.D:
			return self.D
		self.D = Sequential()
		depth = 64
		dropout = 0.4
		# In: 60 x 60 x 1, depth = 1
		# Out: 14 x 14 x 1, depth=64
		self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=self.image_shape, padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(dropout))

		self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(dropout))

		self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(dropout))

		self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
		self.D.add(LeakyReLU(alpha=0.2))
		self.D.add(Dropout(dropout))

		# Out: 1-dim probability
		self.D.add(Flatten())
		self.D.add(Dense(1))
		self.D.add(Activation('sigmoid'))
		self.D.summary()
		return self.D

	def generator(self):
		if self.G:
			return self.G
		self.G = Sequential()
		dropout = 0.4
		depth = 64+64+64+64
		dim = 15
		# In: noise_dim
		# Out: dim x dim x depth
		self.G.add(Dense(dim*dim*depth, input_dim=self.noise_dim))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))
		self.G.add(Reshape((dim, dim, depth)))
		self.G.add(Dropout(dropout))

		# In: dim x dim x depth
		# Out: 2*dim x 2*dim x depth/2
		self.G.add(UpSampling2D())
		self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))

		self.G.add(UpSampling2D())
		self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))

		self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
		self.G.add(BatchNormalization(momentum=0.9))
		self.G.add(Activation('relu'))

		# Out: 60 x 60 x 3 grayscale image [0.0,1.0] per pix
		self.G.add(Conv2DTranspose(3, 5, padding='same'))
		self.G.add(Activation('sigmoid'))
		self.G.summary()
		return self.G

	def discriminator_model(self):
		if self.DM:
			return self.DM
		optimizer = RMSprop(lr=0.0002, decay=6e-8)
		self.DM = Sequential()
		self.DM.add(self.discriminator())
		self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.DM

	def adversarial_model(self):
		if self.AM:
			return self.AM
		optimizer = RMSprop(lr=0.0001, decay=3e-8)
		self.AM = Sequential()
		self.AM.add(self.generator())
		self.AM.add(self.discriminator())
		self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		return self.AM