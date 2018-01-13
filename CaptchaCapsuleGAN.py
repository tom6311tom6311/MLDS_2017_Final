import os
import shutil
from Model import CapsCapsuleGANModel
import Preproccessor
import numpy as np
import matplotlib

# disable display setting
matplotlib.use('Agg')

import matplotlib.pyplot as plt


NOISE_DIM = 100
NUM_CHK_SAMPLES = 16
OURPUT_IMG_DIR = 'output/capsgan/'

class CaptchaCapsuleGAN(object):
	def __init__(self, image_shape):
		self.image_shape = image_shape
		self.Preproccessor = Preproccessor.Preprocessor(image_shape)
		self.CapsCapsuleGANModel = CapsCapsuleGANModel.CapsCapsuleGANModel(image_shape)
		self.discriminator =  self.CapsCapsuleGANModel.discriminator_model()
		self.adversarial = self.CapsCapsuleGANModel.adversarial_model()
		self.generator = self.CapsCapsuleGANModel.generator()

		if os.path.exists(OURPUT_IMG_DIR):
			shutil.rmtree(OURPUT_IMG_DIR)
		os.makedirs(OURPUT_IMG_DIR)

	def train(self, train_steps=2000, batch_size=256, save_interval=0):
		noise_input = None
		if save_interval > 0:
			noise_input = np.random.uniform(-1.0, 1.0, size=[NUM_CHK_SAMPLES, NOISE_DIM])
		for i in range(train_steps):
			images_train, labels_train = self.Preproccessor.loadData(num_to_load=batch_size, shuffle=False)
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, NOISE_DIM])
			images_fake = self.generator.predict(noise)
			x = np.concatenate((images_train, images_fake))
			y = np.ones([2*batch_size, 1])
			y[batch_size:, :] = 0
			d_loss = self.discriminator.train_on_batch(x, y)

			y = np.ones([batch_size, 1])
			noise = np.random.uniform(-1.0, 1.0, size=[batch_size, NOISE_DIM])
			a_loss = self.adversarial.train_on_batch(noise, y)
			log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
			log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
			print(log_mesg)
			if save_interval>0:
				if (i+1)%save_interval==0:
					self.plot_images(save2file=True, samples=NUM_CHK_SAMPLES, noise=noise_input, step=(i+1))

	def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
		if fake:
			if noise is None:
				noise = np.random.uniform(-1.0, 1.0, size=[samples, NOISE_DIM])
			else:
				filename = "fake_%d.png" % step
			images = self.generator.predict(noise)
		else:
			filename = 'real.png'
			images, _ = self.Preproccessor.loadData(num_to_load=samples, shuffle=True)

		plt.figure(figsize=(10,10))
		for i in range(samples):
			plt.subplot(4, 4, i+1)
			image = images[i, :, :, :]
			image = np.reshape(image, self.image_shape)
			plt.imshow(image)
			plt.axis('off')
		plt.tight_layout()
		if save2file:
			plt.savefig(OURPUT_IMG_DIR + filename)
			plt.close('all')
		else:
			plt.show()
