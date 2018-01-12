import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import numpy as np

def plot(samples, filename):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    plt.savefig('out/{}.png'.format(filename.zfill(3)), bbox_inches='tight')
    plt.close(fig)

def save_image_train(epoch, image, args, generated = True):
    im_num = 8*8
    fig = plt.figure(figsize=(8, 8))
    image = image/2. + 0.5
    for i in range(im_num):
        ax = fig.add_subplot(im_num/8, 8, i+1)
        ax.imshow(image[i, :, :, :].reshape(args.img_width,args.img_height,3))
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()

        if generated:
            fig.savefig(args.save_img_dir + '/gen_epoch' + str(epoch) + '.jpg')
        else:
            fig.savefig(args.save_img_dir + '/real_epoch' + str(epoch) + '.jpg')

    plt.close(fig)

def combine_images(generated_images, height=None, width=None):
    # modify from: https://github.com/XifengGuo/CapsNet-Keras/blob/master/utils.py
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1], 3),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1], :] = \
            img[:, :, :]
    return image