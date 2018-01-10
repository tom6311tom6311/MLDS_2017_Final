from captcha.image import ImageCaptcha

LABEL_FILE_NAME = 'label.csv'

class CaptchaGenerator:
  def generateImage(self, phrase, numGen, outputDir, prefix):
    outputDir = outputDir if outputDir[-1] == '/' else outputDir + '/'
    with open(outputDir + LABEL_FILE_NAME, 'a') as labelFile:
      labelFile.write(prefix + ',' + phrase + '\n')
      labelFile.close()
    for i in range(numGen):
      ImageCaptcha().write(phrase, outputDir + prefix + '_' + str(i) + '.png')
