import argparse
import tensorflow as tf
from capsLayer import CapsLayer


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

class GAN:
    def __init__(self, args):
        self.args = args
        self.act = tf.nn.tanh if args.prep else tf.nn.sigmoid
        self.num_class = 26

        self.isTrain = tf.placeholder(tf.bool)
        self.feat_holder = tf.placeholder(tf.float32, [self.args.batch_size, self.num_class])
        with tf.variable_scope('gen'):
            self.noise_holder, self.fake_img = self.generator()

        with tf.variable_scope('dis'):
            self.img_holder = tf.placeholder(tf.float32,
                            [self.args.batch_size, self.args.img_width, self.args.img_height, self.args.channel])
            self.random_feat_holder = tf.placeholder(tf.float32, [self.args.batch_size, self.num_class])
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
        noise_holder = tf.placeholder(tf.float32, [self.args.batch_size, self.args.noise_dim])
        concat = tf.concat([noise_holder, self.feat_holder], axis=1)
        dense1 = tf.layers.dense(concat, 128, activation=tf.nn.relu)
        fake_img = tf.layers.dense(dense1, self.args.img_width*self.args.img_height*self.args.channel, activation=self.act)
        reshape3d = tf.reshape(fake_img, [-1, self.args.img_width, self.args.img_height, self.args.channel])
        return noise_holder, reshape3d

    def discriminator(self, input, feat, reuse):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        reshape = tf.reshape(input,
                    [-1, self.args.img_width*self.args.img_height*self.args.channel])
        concat = tf.concat([reshape, feat], axis=1)
        dense1 = tf.layers.dense(concat, 128, activation=tf.nn.relu, name='dense1')
        logits = tf.layers.dense(dense1, 1, activation=None, name='S_pred')
        return logits

class DCGAN(GAN):
    def generator(self):
        isTrain = self.isTrain
        noise_holder = tf.placeholder(tf.float32, [None, self.args.noise_dim])
        concat_feat = tf.concat([noise_holder, self.feat_holder], axis=1)
        dense1 = tf.layers.dense(concat_feat, self.args.img_width*self.args.img_height*self.args.channel)
        reshape_noise = tf.reshape(dense1, [-1, self.args.img_width//4, self.args.img_height//4,
                                            self.args.channel*16])
        lrelu0 = lrelu(tf.layers.batch_normalization(reshape_noise, training=isTrain))

        conv1 = tf.layers.conv2d_transpose(lrelu0, 32, [5, 5], strides=(2, 2), padding='same')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain))

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, self.args.channel, [5, 5], strides=(2, 2), padding='same')
        o = self.act(conv2)
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

        flat = tf.reshape(conv2, [-1, self.args.img_width*self.args.img_height])
        concat_feat = tf.concat([flat, feat], axis=1)
        dense2 = tf.layers.dense(concat_feat, self.args.img_width*self.args.img_height
                                 , activation=lrelu, name='dense2')
        feat_logits = tf.layers.dense(dense2, 1, name='feat_logits')

        return feat_logits

class DCCapsGAN(GAN):
    def generator(self):
        isTrain = self.isTrain
        noise_holder = tf.placeholder(tf.float32, [None, self.args.noise_dim])
        concat_feat = tf.concat([noise_holder, self.feat_holder], axis=1)
        dense1 = tf.layers.dense(concat_feat, self.args.img_width*self.args.img_height*self.args.channel)
        reshape_noise = tf.reshape(dense1, [-1, self.args.img_width//4, self.args.img_height//4,
                                            self.args.channel*16])
        lrelu0 = lrelu(tf.layers.batch_normalization(reshape_noise, training=isTrain))

        conv1 = tf.layers.conv2d_transpose(lrelu0, 32, [5, 5], strides=(2, 2), padding='same')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain))

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, self.args.channel, [5, 5], strides=(2, 2), padding='same')
        o = self.act(conv2)
        return noise_holder, o

    def discriminator(self, input, feat, reuse):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        isTrain = self.isTrain
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(input, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            #assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

        # 1st hidden layer
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
            # assert caps1.get_shape() == [cfg.batch_size, 1152, 8, 1]
        # 2nd hidden layer
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=self.num_class, vec_len=16, with_routing=True, layer_type='FC')
            caps2 = digitCaps(caps1)
            # output shape = [cfg.batch_size, self.num_class, 16, 1]
        
        res_caps = tf.reshape(caps2, [-1, self.num_class, 16])
        res_feat = tf.reshape(feat, [-1, 1, self.num_class])
        res_flat = tf.matmul(res_feat, res_caps)
        # flat = tf.reshape(caps2, [-1, 16*self.num_class])
        flat = tf.reshape(res_flat, [-1, 16])
        '''
        conv1 = tf.layers.conv2d(input, 32, 5, strides=(2, 2), padding='same', name='conv1')
        lrelu1 = lrelu(conv1)
        conv2 = tf.layers.conv2d(conv1, 16, 5, strides=(2, 2), padding='same', name='conv2')
        flat = tf.reshape(conv2, [-1, self.args.img_width*self.args.img_height])
        '''
        # concat_feat = tf.concat([flat, feat], axis=1)
        # dense2 = tf.layers.dense(flat, self.args.img_width*self.args.img_height
        #                          , activation=lrelu, name='dense2')
        feat_logits = tf.layers.dense(flat, 1, name='feat_logits')

        return feat_logits

class ACGAN(object):
    def __init__(self, args):
        self.args = args
        self.act = tf.nn.tanh if args.prep else tf.nn.sigmoid
        self.num_class = 26

        self.isTrain = tf.placeholder(tf.bool)
        self.feat_holder = tf.placeholder(tf.float32, [self.args.batch_size, self.num_class])
        with tf.variable_scope('gen'):
            self.noise_holder, self.fake_img = self.generator()

        with tf.variable_scope('dis'):
            self.img_holder = tf.placeholder(tf.float32,
                            [self.args.batch_size, self.args.img_width, self.args.img_height, self.args.channel])
            self.random_feat_holder = tf.placeholder(tf.float32, [self.args.batch_size, self.num_class])
            self.rr_logit, self.rr_class =\
                self.discriminator(self.img_holder, self.feat_holder, None)

            self.fr_logit, self.fr_class =\
                self.discriminator(self.fake_img, self.feat_holder, True)

            self.rf_logit, _ =\
                self.discriminator(self.img_holder, self.random_feat_holder, True)

            self.ff_logit, _ =\
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

        def softmax_cross_entropy(logit, label):
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))

        def sigmoid_cross_entropy(logit, label):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))

        self.rr_loss = sigmoid_cross_entropy(self.rr_logit, tf.ones([args.batch_size, 1]))
        self.fr_loss = sigmoid_cross_entropy(self.fr_logit, tf.zeros([args.batch_size, 1]))
        self.rf_loss = sigmoid_cross_entropy(self.rf_logit, tf.zeros([args.batch_size, 1]))
        self.ff_loss = sigmoid_cross_entropy(self.rr_logit, tf.zeros([args.batch_size, 1]))

        self.rc_loss = softmax_cross_entropy(self.rr_class, self.feat_holder)
        self.fc_loss = softmax_cross_entropy(self.fr_class, self.feat_holder)

        self.D_loss_S = (self.rr_loss + self.fr_loss + self.rf_loss + self.rr_loss)/2 
        self.D_loss = self.D_loss_S + self.rc_loss+self.fc_loss
        self.dis_optimizer = tf.train.AdamOptimizer(self.args.lr*3)
        self.opt_dis = self.dis_optimizer.minimize(self.D_loss, var_list=dis_vars)

        self.G_loss_S = sigmoid_cross_entropy(self.fr_logit, tf.ones([args.batch_size, 1]))
        self.G_loss = self.G_loss_S + self.rc_loss+self.fc_loss
        self.gen_optimizer = tf.train.AdamOptimizer(self.args.lr)
        self.opt_gen = self.gen_optimizer.minimize(self.G_loss, var_list=gen_vars)

    def generator(self):
        isTrain = self.isTrain
        noise_holder = tf.placeholder(tf.float32, [None, self.args.noise_dim])
        concat_feat = tf.concat([noise_holder, self.feat_holder], axis=1)
        dense1 = tf.layers.dense(concat_feat, self.args.img_width*self.args.img_height*self.args.channel)
        reshape_noise = tf.reshape(dense1, [-1, self.args.img_width//4, self.args.img_height//4,
                                            self.args.channel*16])
        lrelu0 = lrelu(tf.layers.batch_normalization(reshape_noise, training=isTrain))

        conv1 = tf.layers.conv2d_transpose(lrelu0, 32, [5, 5], strides=(2, 2), padding='same')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain))

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, self.args.channel, [5, 5], strides=(2, 2), padding='same')
        o = self.act(conv2)
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

        flat = tf.reshape(conv2, [-1, self.args.img_width*self.args.img_height])
        dense1 = tf.layers.dense(flat, self.num_class, name='dense1')
        concat_feat = tf.concat([flat, feat], axis=1)
        dense2 = tf.layers.dense(concat_feat, self.args.img_width*self.args.img_height
                                 , activation=lrelu, name='dense2')
        feat_logits = tf.layers.dense(dense2, 1, name='feat_logits')

        return feat_logits, dense1

class DCCapsACGAN(object):
    def __init__(self, args):
        self.args = args
        self.act = tf.nn.tanh if args.prep else tf.nn.sigmoid
        self.num_class = 26

        self.isTrain = tf.placeholder(tf.bool)
        self.feat_holder = tf.placeholder(tf.float32, [self.args.batch_size, self.num_class])
        with tf.variable_scope('gen'):
            self.noise_holder, self.fake_img = self.generator()

        with tf.variable_scope('dis'):
            self.img_holder = tf.placeholder(tf.float32,
                            [self.args.batch_size, self.args.img_width, self.args.img_height, self.args.channel])
            self.random_feat_holder = tf.placeholder(tf.float32, [self.args.batch_size, self.num_class])
            self.rr_logit, self.rr_class =\
                self.discriminator(self.img_holder, self.feat_holder, None)

            self.fr_logit, self.fr_class =\
                self.discriminator(self.fake_img, self.feat_holder, True)

            self.rf_logit, _ =\
                self.discriminator(self.img_holder, self.random_feat_holder, True)

            self.ff_logit, _ =\
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

        def softmax_cross_entropy(logit, label):
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))

        def sigmoid_cross_entropy(logit, label):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))

        self.rr_loss = sigmoid_cross_entropy(self.rr_logit, tf.ones([args.batch_size, 1]))
        self.fr_loss = sigmoid_cross_entropy(self.fr_logit, tf.zeros([args.batch_size, 1]))
        self.rf_loss = sigmoid_cross_entropy(self.rf_logit, tf.zeros([args.batch_size, 1]))
        self.ff_loss = sigmoid_cross_entropy(self.rr_logit, tf.zeros([args.batch_size, 1]))

        self.rc_loss = softmax_cross_entropy(self.rr_class, self.feat_holder)
        self.fc_loss = softmax_cross_entropy(self.fr_class, self.feat_holder)

        self.D_loss_S = (self.rr_loss + self.fr_loss + self.rf_loss + self.rr_loss)/2 
        self.D_loss = self.D_loss_S + self.rc_loss+self.fc_loss
        self.dis_optimizer = tf.train.AdamOptimizer(self.args.lr*3)
        self.opt_dis = self.dis_optimizer.minimize(self.D_loss, var_list=dis_vars)

        self.G_loss_S = sigmoid_cross_entropy(self.fr_logit, tf.ones([args.batch_size, 1]))
        self.G_loss = self.G_loss_S + self.rc_loss+self.fc_loss
        self.gen_optimizer = tf.train.AdamOptimizer(self.args.lr)
        self.opt_gen = self.gen_optimizer.minimize(self.G_loss, var_list=gen_vars)

    def generator(self):
        isTrain = self.isTrain
        noise_holder = tf.placeholder(tf.float32, [None, self.args.noise_dim])
        concat_feat = tf.concat([noise_holder, self.feat_holder], axis=1)
        dense1 = tf.layers.dense(concat_feat, self.args.img_width*self.args.img_height*self.args.channel)
        reshape_noise = tf.reshape(dense1, [-1, self.args.img_width//4, self.args.img_height//4,
                                            self.args.channel*16])
        lrelu0 = lrelu(tf.layers.batch_normalization(reshape_noise, training=isTrain))

        conv1 = tf.layers.conv2d_transpose(lrelu0, 32, [5, 5], strides=(2, 2), padding='same')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain))

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, self.args.channel, [5, 5], strides=(2, 2), padding='same')
        o = self.act(conv2)
        return noise_holder, o

    def discriminator(self, input, feat, reuse):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        isTrain = self.isTrain
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            conv1 = tf.contrib.layers.conv2d(input, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            #assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

        # 1st hidden layer
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
            # assert caps1.get_shape() == [cfg.batch_size, 1152, 8, 1]
        # 2nd hidden layer
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=self.num_class, vec_len=16, with_routing=True, layer_type='FC')
            caps2 = digitCaps(caps1)
            # output shape = [cfg.batch_size, self.num_class, 16, 1]
        
        caps_length = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True) + 1e-9)
        flat_caps_length = tf.reshape(caps_length, [-1, self.num_class])

        res_caps = tf.reshape(caps2, [-1, self.num_class, 16])
        res_feat = tf.reshape(feat, [-1, 1, self.num_class])
        res_flat = tf.matmul(res_feat, res_caps)
        flat = tf.reshape(res_flat, [-1, 16])
        feat_logits = tf.layers.dense(flat, 1, name='feat_logits')

        return feat_logits, flat_caps_length

