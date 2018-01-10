import os
import shutil
import argparse
import nltk
import CaptchaGenerator
import numpy as np
from nltk.corpus import words

nltk.download('words')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Raw Captcha Image Generator")
	parser.add_argument('--type', default='digit', type=str,
			help="Type of phrase, can be 'digit' or 'char'")
	parser.add_argument('--num', default=1, type=int,
			help="Number of phrases")
	parser.add_argument('--num_word', default=1, type=int,
			help="Number of words each phrase")
	parser.add_argument('--len_word', default=5, type=int,
			help="Length of each word")
	parser.add_argument('--num_gen', default=5, type=int,
			help="Number of generations each phrase")
	parser.add_argument('--output', default='raw/', type=str,
			help="Output folder")
	args = parser.parse_args()

	if os.path.exists(args.output):
		shutil.rmtree(args.output)
	os.makedirs(args.output)

	captchaGen = CaptchaGenerator.CaptchaGenerator()

	for i in range(args.num):
		if args.type == 'digit':
			words_gen = map(str, np.random.choice(np.power(10,args.len_word), args.num_word))
		else:
			corpus = words.words()
			words_gen = []
			while len(words_gen) < args.num_word:
				word = np.random.choice(corpus)
				if len(word) <= args.len_word:
					words_gen.append(word)
		phrase = ' '.join(words_gen)
		captchaGen.generateImage(phrase, args.num_gen, args.output, str(i))
