import os
import shutil
import random
import Preproccessor
import tensorflow as tf
import numpy as np
from argument import parse_args
from model import DCGAN as GAN
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import to_categorical
from utils import save_image_train

LOAD_FROM_MNIST = False

args = parse_args()

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
    if args.clean:
        if os.path.exists(args.save_img_dir):
            shutil.rmtree(args.save_img_dir)
        if os.path.exists(args.log_dir):
            shutil.rmtree(args.log_dir)
    if not os.path.exists(args.save_img_dir):
        os.mkdir(args.save_img_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)



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

        saver = sv.saver(max_to_keep=1000)
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
                    print('[n_epoch: %d, D_loss: %f, G_loss: %f]' % (n_epoch, D_loss_curr, G_loss_curr))
                    label = np.argmax(batch_feat[0])
                    filename = str(n_epoch)+'_'+str(label)+'.jpg'
                    misc.imsave(os.path.join(args.save_img_dir, filename), fake_img[0, :, :, :])
                    save_image_train(n_epoch, fake_img, args, generated = True)
                    save_image_train(n_epoch, batch_img, args, generated = False)
                    # save_path = saver.save(sess, args.log_dir+'/model_'+str(n_epoch)+'.ckpt')
                    # print("Model saved in file: %s" % save_path)
                    saver.save(sess, save_path=args.log_dir, global_step=n_epoch)
