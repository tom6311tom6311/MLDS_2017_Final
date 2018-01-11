import tensorflow as tf
import numpy as np
import random
from argument import parse_args
from model import DCGAN as GAN
import os
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data
import Preproccessor
from keras.utils import to_categorical

LOAD_FROM_MNIST = False

args = parse_args()

if LOAD_FROM_MNIST:
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
else:
    prep = Preproccessor.Preprocessor(image_shape=[args.img_width,args.img_height,3])

def process_img(img):
    return (img-0.5)/0.5

def get_batch():
    if LOAD_FROM_MNIST:
        img, label = mnist.train.next_batch(args.batch_size)
        resized = np.reshape(img, [args.batch_size, 28, 28, 1])
        resized = process_img(resized)
        return img, label
    else:
        img, txt_label = prep.loadData(args.batch_size, True)
        label = to_categorical(txt_label.astype('float32'))
        return process_img(img), label

def get_noise():
    return np.random.uniform(-1., 1., [args.batch_size, args.noise_dim])

def get_random_feat():
    a = np.random.randint(0, 10, [args.batch_size])
    random_feat = np.zeros([args.batch_size, 10])
    for i in range(args.batch_size):
        random_feat[i, a[i]] = 1
    return random_feat

if __name__ == '__main__':
    if not os.path.exists(args.save_img_dir):
        os.mkdir(args.save_img_dir)
    with tf.Graph().as_default() as graph:
        initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
        with tf.variable_scope('model', reuse=None, initializer=initializer) as scope:
            model = GAN(args)
            scope.reuse_variables()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level =\
        tf.OptimizerOptions.ON_1
        sv = tf.train.Supervisor(logdir=args.log_dir,
                             save_model_secs=args.save_model_secs)

        with sv.managed_session(config=config) as sess:
            for n_epoch in range(args.max_epoch):
                batch_noise = get_noise()
                batch_img, batch_feat = get_batch()
                random_feat = get_random_feat()
                _, D_loss_curr = sess.run([model.opt_dis, model.D_loss],
                                          feed_dict={model.noise_holder: batch_noise,
                                                     model.feat_holder: batch_feat,
                                                     model.img_holder: batch_img,
                                                     model.random_feat_holder: random_feat,
                                                     model.isTrain: True})

                _, G_loss_curr, fake_img = sess.run([model.opt_gen, model.G_loss, model.fake_img],
                                                    feed_dict={model.noise_holder: batch_noise,
                                                               model.feat_holder: batch_feat,
                                                               model.isTrain: True})

                if (n_epoch % args.info_epoch == 0):
                    print('n_epoch: ', n_epoch)
                    print('D_loss:', D_loss_curr)
                    print('G_loss:', G_loss_curr)
                    label = np.argmax(batch_feat[0])
                    filename = str(n_epoch)+'_'+str(label)+'.jpg'
                    misc.imsave(os.path.join(args.save_img_dir, filename), fake_img[0, :, :, 0])
