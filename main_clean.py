import csv
import os
from pydub import AudioSegment
import argparse
import sys
from loguru import logger

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", dest="input_dir", help="The directory containing a wavs folder and the metadata.csv.", type=str,
		default="E:/Datasets3/output_train/de_DE/by_book/mix/mix/various/")
	parser.add_argument("--cleansed_medatata_file", dest="cleansed_medatata_file", help="The name of the new metadata.csv in input_dir.", type=str,
		default="metadata2.csv")
	parser.add_argument("--min_sentence_length", dest="min_sentence_length", help="Minimum of characters in a sentence.", type=int, default=5)
	parser.add_argument("--min_audio_length", dest="min_audio_length", help="Minimum length of the generated audio in seconds.", type=int, default=1)
	parser.add_argument("--max_audio_length", dest="max_audio_length", help="Maximum length of the generated audio in seconds.", type=int, default=30)
	args = parser.parse_args()
	
	meta_csv = os.path.join(args.input_dir, "metadata.csv")
	new_meta_csv = os.path.join(args.input_dir, args.cleansed_medatata_file)
	
	if os.path.exists(new_meta_csv):
		logger.error("The cleansed_metadata_file {} exists. Exiting.", args.cleansed_medatata_file)
		sys.exit()
	
	with open(new_meta_csv, 'w', encoding='utf8', newline='') as f:
		writer = csv.writer(f, delimiter ='|')
		csv_reader = csv.reader(open(meta_csv, encoding="utf8"), delimiter="|", quoting=csv.QUOTE_NONE)
		meta_csv = list(csv_reader)
		for entry in meta_csv:
			wav_file = entry[0]
			text = entry[1]
			
			audio = AudioSegment.from_wav(os.path.join(args.input_dir, "wavs", wav_file + ".wav"))
			minCharLength = len(text) >= args.min_sentence_length
			maxAudioLength = audio.duration_seconds <= args.max_audio_length
			minAudioLength = audio.duration_seconds >= args.min_audio_length
			
			if minCharLength and maxAudioLength and minAudioLength:
				writer.writerow([wav_file, text])
			else:
				logger.info("Audio file {}.wav with content {} does not fit with a length of {} seconds.", wav_file, text, audio.duration_seconds)
				mp3file = os.path.join(args.input_dir, "wavs", wav_file + ".wav")
				txtfile = os.path.join(args.input_dir, "wavs", wav_file + ".txt")
				if os.path.exists(mp3file):
					os.rename(mp3file, mp3file + ".backup")
				if os.path.exists(txtfile):
					os.rename(txtfile, txtfile + ".backup")
		