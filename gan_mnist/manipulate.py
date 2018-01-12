# manipulate latent
import tensorflow as tf
import numpy as np
from argument import parse_args
from model import DCGAN as GAN
import os
from PIL import Image
from utils import combine_images

if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.save_manipulate_dir):
        os.mkdir(args.save_manipulate_dir)

    with tf.variable_scope('model') as scope:
        model = GAN(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config = config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, args.log_dir + '/model.ckpt')

        # manipulate latent
        # mpdify from: https://github.com/XifengGuo/CapsNet-Keras/blob/master/capsulenet.py
        feat = np.zeros([1, 10])
        feat[0, args.digit] = 1
        noise = np.zeros([1, args.noise_dim])
        dimension = range(0, args.noise_dim, int(args.noise_dim/10))
        x_recons = []
        for dim in list(dimension):
            for r in [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]:
                tmp = np.copy(noise)
                tmp[:, dim] = r
                x_recon = sess.run([model.fake_img],feed_dict={model.noise_holder: tmp,
                                                               model.feat_holder: feat,
                                                               model.isTrain: False})
                x_recons.append(x_recon)

        x_recons = np.concatenate(x_recons)
        x_recons = np.reshape(x_recons, [x_recons.shape[0], args.img_width,args.img_height,args.channel])

        img = combine_images(x_recons, height=10)
        image = (img/2. + 0.5)*255
        Image.fromarray(image.astype(np.uint8)).save(args.save_manipulate_dir + '/manipulate-%d.png' % args.digit)



		