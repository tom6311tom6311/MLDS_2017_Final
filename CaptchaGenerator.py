from captcha.image import ImageCaptcha

LABEL_FILE_NAME = 'label.csv'

class CaptchaGenerator:
  def __init__(self, width, height):
    self.width = width
    self.height = height

  def generateImage(self, phrase, numGen, outputDir, prefix):
    outputDir = outputDir if outputDir[-1] == '/' else outputDir + '/'
    with open(outputDir + LABEL_FILE_NAME, 'a') as labelFile:
      labelFile.write(prefix + ',' + phrase + '\n')
      labelFile.close()
    for i in range(numGen):
      ImageCaptcha(self.width, self.height).write(phrase, outputDir + prefix + '_' + str(i) + '.png')
