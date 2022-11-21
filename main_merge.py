import os
from tqdm import tqdm
import sys
from loguru import logger
import argparse

			
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--files', help='Comma separated list of files.', type=str)
	parser.add_argument('-o', '--output', help='File that all files are merged to.', type=str)
	args = parser.parse_args()
	logger.remove()
	logger.add(sys.stderr, level="DEBUG")
	files = [f for f in args.files.split(',')]
	
	# calculate a random prefix to avoid overwriting files
	prefix = ''.join(random.choice(string.ascii_lowercase) for i in range(3))

	out_file = args.output
	counter = 0
	with open(metadata_file, 'w', encoding='utf8') as outfile:
		writer = csv.writer(outfile, delimiter ='|')
		
		for f in args.files:
			with open(f, 'r', encoding='utf8') as infile:
				csv_reader = csv.reader(infile, delimiter="|", quoting=csv.QUOTE_NONE)
				meta_csv = list(csv_reader)
				for entry in meta_csv:
					wav_file = entry[0]
					text = entry[1]
					wav_file_path = os.path.join(args.input_dir, "wavs", wav_file + ".wav")
					if os.exists(wav_file_path):
						writer.writerow([prefix + str(counter), text])
						os.rename(wav_file_path, os.path.join(args.input_dir, "wavs", prefix+str(counter) + ".wav")
						counter += 1
	logger.info("Done")
