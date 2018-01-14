import tensorflow as tf
import numpy as np
import random
from argument import parse_args
from model import DCCapsGAN as CapsGAN
import os
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data
import Preproccessor
from keras.utils import to_categorical
from utils import save_image_train, save_image_train_by_digit

import logging
def myLog(msg, epoch = -1):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.log(level=logging.DEBUG, msg=('Epoch['+str(epoch)+'] '+str(msg)))

LOAD_FROM_MNIST = False

args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if LOAD_FROM_MNIST:
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
else:
    prep = Preproccessor.Preprocessor(image_shape=[args.img_width,args.img_height,3])

def process_img(img):
    if args.prep:
        return (img-0.5)/0.5
    else:
        return img

def get_batch():
    if LOAD_FROM_MNIST:
        img, label = mnist.train.next_batch(args.batch_size)
        resized = np.reshape(img, [args.batch_size, 28, 28, 1])
        resized = process_img(resized)
        return img, label
    else:
        img, txt_label = prep.loadData(args.batch_size, True)
        label = to_categorical(txt_label.astype('float32'), num_classes=10)
        return process_img(img), label

def get_noise():
    return np.random.uniform(-1., 1., [args.batch_size, args.noise_dim])

def get_random_feat(true_feat):
    true_a = np.argmax(true_feat, axis=1)
    a = np.random.randint(1, 10, [args.batch_size])
    a = (true_a + a)%10
    random_feat = np.zeros([args.batch_size, 10])
    for i in range(args.batch_size):
        random_feat[i, a[i]] = 1
    return random_feat

if __name__ == '__main__':
    if not os.path.exists(args.save_img_dir):
        os.mkdir(args.save_img_dir)

    with tf.Graph().as_default() as graph:
        initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
        with tf.variable_scope('model_capsule', reuse=None, initializer=initializer) as scope:
            model = CapsGAN(args)
            scope.reuse_variables()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level =\
        tf.OptimizerOptions.ON_1
        sv = tf.train.Supervisor(logdir=args.log_dir,
                             save_model_secs=args.save_model_secs)

        saver = sv.saver
        with sv.managed_session(config=config) as sess:
            save_noise = np.random.uniform(-1., 1., [10, args.noise_dim])
            save_feat = to_categorical(np.arange(10), num_classes=10)
            save_noise_fill = np.concatenate((save_noise, np.zeros((args.batch_size-10, args.noise_dim))), axis=0)
            save_feat_fill  = np.concatenate((save_feat,  np.zeros((args.batch_size-10, 10))), axis=0)
            
            save_img_fill = sess.run([model.fake_img], feed_dict={model.noise_holder: save_noise_fill,
                                                        model.feat_holder: save_feat_fill,
                                                        model.isTrain: True})
            save_img = save_img_fill[0][0:10, :, :, :]
            save_image_train_by_digit('_test', save_img, args, generated = True)

