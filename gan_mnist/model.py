import argparse
import tensorflow as tf


def lrelu(x, th=0.01):
    return tf.maximum(th * x, x)

class GAN:
    def __init__(self, args):
        self.args = args

        self.isTrain = tf.placeholder(tf.bool)
        self.feat_holder = tf.placeholder(tf.float32, [None, 10])
        with tf.variable_scope('gen'):
            self.noise_holder, self.fake_img = self.generator()

        with tf.variable_scope('dis'):
            self.img_holder = tf.placeholder(tf.float32, [None, 28, 28, 1])
            self.random_feat_holder = tf.placeholder(tf.float32, [None, 10])
            self.rr_logit =\
                self.discriminator(self.img_holder, self.feat_holder, None)

            self.fr_logit =\
                self.discriminator(self.fake_img, self.feat_holder, True)

            self.rf_logit =\
                self.discriminator(self.img_holder, self.random_feat_holder, True)

            self.ff_logit =\
                self.discriminator(self.fake_img, self.random_feat_holder, True)

        gen_vars = []
        dis_vars = []
        t_vars = tf.trainable_variables()
        for var in t_vars:
            if "gen" in var.name:
                gen_vars.append(var)
            elif "dis" in var.name:
                dis_vars.append(var)
        for var in gen_vars:
            print(var)
        print('')
        for var in dis_vars:
            print(var)

        def cross_entropy(logit, label):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))

        self.rr_loss = cross_entropy(self.rr_logit, tf.ones([args.batch_size, 1]))
        self.fr_loss = cross_entropy(self.fr_logit, tf.zeros([args.batch_size, 1]))
        self.rf_loss = cross_entropy(self.rf_logit, tf.zeros([args.batch_size, 1]))
        self.ff_loss = cross_entropy(self.rr_logit, tf.zeros([args.batch_size, 1]))

        self.D_loss = self.rr_loss + self.fr_loss + self.rf_loss + self.rr_loss
        self.dis_optimizer = tf.train.AdamOptimizer(self.args.lr*3)
        self.opt_dis = self.dis_optimizer.minimize(self.D_loss, var_list=dis_vars)

        self.G_loss_S = cross_entropy(self.fr_logit, tf.ones([args.batch_size, 1]))
        self.G_loss = self.G_loss_S
        self.gen_optimizer = tf.train.AdamOptimizer(self.args.lr)
        self.opt_gen = self.gen_optimizer.minimize(self.G_loss, var_list=gen_vars)


    def generator(self):
        noise_holder = tf.placeholder(tf.float32, [None, self.args.noise_dim])
        concat = tf.concat([noise_holder, self.feat_holder], axis=1)
        dense1 = tf.layers.dense(concat, 128, activation=tf.nn.relu)
        fake_img = tf.layers.dense(dense1, 784, activation=tf.nn.tanh)
        reshape3d = tf.reshape(fake_img, [-1, 28, 28, 1])
        return noise_holder, reshape3d

    def discriminator(self, input, feat, reuse):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        reshape = tf.reshape(input, [-1, 784])
        concat = tf.concat([reshape, feat], axis=1)
        dense1 = tf.layers.dense(concat, 128, activation=tf.nn.relu, name='dense1')
        logits = tf.layers.dense(dense1, 1, activation=None, name='S_pred')
        return logits

class DCGAN(GAN):
    def generator(self):
        isTrain = self.isTrain
        noise_holder = tf.placeholder(tf.float32, [None, self.args.noise_dim])
        concat_feat = tf.concat([noise_holder, self.feat_holder], axis=1)
        dense1 = tf.layers.dense(concat_feat, 784)
        reshape_noise = tf.reshape(dense1, [-1, 7, 7, 16])
        lrelu0 = lrelu(tf.layers.batch_normalization(reshape_noise, training=isTrain))

        conv1 = tf.layers.conv2d_transpose(lrelu0, 32, [5, 5], strides=(2, 2), padding='same')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain))

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 1, [5, 5], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv2)
        return noise_holder, o

    def discriminator(self, input, feat, reuse):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        isTrain = self.isTrain
        # 1st hidden layer
        conv1 = tf.layers.conv2d(input, 32, 5, strides=(2, 2), padding='same', name='conv1')
        lrelu1 = lrelu(conv1)
        # 2nd hidden layer
        conv2 = tf.layers.conv2d(conv1, 16, 5, strides=(2, 2), padding='same', name='conv2')

        flat = tf.reshape(conv2, [-1, 784])
        concat_feat = tf.concat([flat, feat], axis=1)
        dense2 = tf.layers.dense(concat_feat, 784, activation=lrelu, name='dense2')
        feat_logits = tf.layers.dense(dense2, 1, name='feat_logits')

        return feat_logits
