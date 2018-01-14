import os
import numpy as np
from scipy import misc

IMG_DIR = '../raw_char/'
IMG_FILE_TYPE = '.png'
LABEL_FILE_NAME = 'label.csv'


class Preprocessor:
  def __init__(self, image_shape):
    self.image_shape = image_shape
    self.label_dict = self.loadLabelDict()
    self.image_files, self.image_labels = self.loadImageFilesAndLabels()
    self.current_loaded_idx = 0

  def loadLabelDict(self):
    label_dict = {}
    with open(IMG_DIR + LABEL_FILE_NAME, 'r') as label_file:
      raw_labels = label_file.readlines()
      for raw_label in raw_labels:
        [prefix,label] = raw_label[:-1].split(',')
        label_dict[prefix] = float(ord(label)-65)
      label_file.close()
    return label_dict

  def loadImageFilesAndLabels(self):
    image_files = []
    image_labels = []
    for f in os.listdir(IMG_DIR):
      if f.endswith(IMG_FILE_TYPE):
        image_files.append(f)
        prefix = f.split('_')[0]
        label = self.label_dict[prefix]
        image_labels.append(label)
    return image_files, image_labels

  def loadData(self, num_to_load=-1, shuffle=False):
    num_all_images = len(self.image_files)
    if (num_to_load == -1):
      # load all
      chosed_idx = np.arange(num_all_images)
    elif (shuffle == True):
      chosed_idx = np.random.choice(num_all_images, num_to_load)
    else:
      chosed_idx = np.remainder(np.arange(self.current_loaded_idx, self.current_loaded_idx + num_to_load), num_all_images)
      self.current_loaded_idx = (self.current_loaded_idx + num_to_load) % num_all_images
    return self.loadImgAndLabel(chosed_idx)

  def loadImgAndLabel(self, idx_to_load):
    images = []
    labels = []
    for idx in idx_to_load:
      images.append(self.loadAndPreproccessImg(self.image_files[idx]))
      labels.append(self.image_labels[idx])
    return np.array(images), np.array(labels)

  def loadAndPreproccessImg(self, img_file_name):
    image = misc.imread(IMG_DIR + img_file_name)
    image = misc.imresize(image, self.image_shape[0:2])
    image = image.astype('float32') / 255
    return image

  # def saveImage(self, name, image, outputPath, feature):
  #   if feature is not None:
  #     desc = self.featToDesc(feature[-self.featDim:])
  #     with open(outputPath + PREDICTION_ABSTRACTS_FILE_NAME, 'a') as abstractFile:
  #       abstractFile.write(str(name) + ', ' + desc + '\n')
  #       abstractFile.close()
  #   image *= 255
  #   misc.imsave(outputPath + str(name) + IMG_FILE_TYPE, image)
