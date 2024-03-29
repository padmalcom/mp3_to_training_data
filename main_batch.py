import vosk
import wave
import json
from pydub import AudioSegment
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
from spacy.lang.de import German
from tqdm import tqdm
import math
from difflib import SequenceMatcher
import copy
import sys
from loguru import logger
from timeit import default_timer as timer
from transformers import pipeline
import argparse
from recasepunc import CasePuncPredictor
import glob
from multiprocessing import Process
import multiprocessing

# This can be a word or a sentence
class AnnotatedSequence:
	def __init__(self, sequence, start, end, speaker, confidence, min_confidence):
		self.sequence = sequence
		self.start = start
		self.end = end
		self.speaker = speaker
		self.confidence = confidence
		self.min_confidence = min_confidence
		
	def get_formated_start(self):
		return self.timeString(self.start)
		
	def get_formated_end(self):
		return self.timeString(self.end)
		
	def timeString(self, seconds):
		minutes = seconds / 60
		seconds = seconds % 60
		hours = int(minutes / 60)
		minutes = int(minutes % 60)
		return '%i:%02i:%06.3f' % (hours, minutes, seconds)
		
	def __repr__(self):
		return "(sequence: {}, start: {}, end: {}, speaker: {}, confidence: {}, min_confidence: {}".format(self.sequence, self.get_formated_start(), self.get_formated_end(), self.speaker, self.confidence, self.min_confidence)

class Transcriber:
	
	def __init__(self):
		vosk.SetLogLevel(-2)
		
		# model_path = 'vosk-model-small-de-0.15'
		model_path = 'vosk-model-de-0.21'
		self.COSINE_DIST = 0.4
		self.sample_rate = 16000
		self.model = vosk.Model(model_path)
		self.rec = vosk.KaldiRecognizer(self.model, self.sample_rate)
		
		punc_predict_path = os.path.abspath('vosk-recasepunc-de-0.21/checkpoint')
		self.casePuncPredictor = CasePuncPredictor(punc_predict_path, lang="de")
		
		logger.info("Initialization done.")
		
	def repair_text(self, text):
		tokens = list(enumerate(self.casePuncPredictor.tokenize(text)))
		text = ''
		for token, case_label, punc_label in self.casePuncPredictor.predict(tokens, lambda x: x[1]):
			mapped = self.casePuncPredictor.map_punc_label(self.casePuncPredictor.map_case_label(token[1], case_label), punc_label)
			logger.trace("Token {}, case_label {}, punc_label {}, mapped {}", token, case_label, punc_label, mapped)

			if token[1].startswith('##'):
				text += mapped.replace(" ", "")
			else:
				text +=  ' ' + mapped.replace(" ", "")
		return text.strip()
		
	def get_words_from_text(self, audio_file):
		rec_results = []
		self.rec = vosk.KaldiRecognizer(self.model, self.sample_rate)
		self.rec.SetWords(True)
		wf = wave.open(audio_file, "rb")
		#for i in tqdm(range(0, math.ceil(wf.getnframes() / 4000)), desc="Discovering audio sequence ..."):
		for i in range(0, math.ceil(wf.getnframes() / 4000)):
		#while True:
		   data = wf.readframes(4000)
		   if len(data) == 0:
			   break
		   if self.rec.AcceptWaveform(data):
			   rec_results.append(self.rec.Result())
		rec_results.append(self.rec.FinalResult())
		wf.close()
		word_results = []
		full_text = ""
		for i, res in enumerate(rec_results):
			words = json.loads(res).get('result')
			if not words:
				continue
			for w in words:
				word_results.append(AnnotatedSequence(w['word'], w['start'], w['end'], None, w['conf'], w['conf']))
				logger.trace("Found new word {}.", word_results[len(word_results)-1])
				full_text += ' ' + w['word']
				
		return word_results, full_text.strip()
				
	def best_sentence_fit(self, unrepaired_words, unrepaired_start_index, repaired_words): # new
		
		punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
		
		repaired_words_without_punct = [w for w in repaired_words if not w in punctuation]

		repaired_sentence_without_punct = ' '.join(repaired_words_without_punct)

		# create i sentences
		best_ratio = 0
		best_end_index = 0
		for i in range(unrepaired_start_index, len(unrepaired_words)):
			sub_list = [x.sequence for x in unrepaired_words[unrepaired_start_index:i+1]]
			sentence_to_compare = ' '.join(sub_list)
			ratio = SequenceMatcher(None, repaired_sentence_without_punct, sentence_to_compare).ratio()
			logger.trace("{}: sentence to compare: {} ratio: {}", i, sentence_to_compare, ratio)
			if ratio > best_ratio:
				best_ratio = ratio
				best_end_index = i
				
			# Experimental - don't compare sentences, if the length of the constructed length is twice the size
			if len(repaired_sentence_without_punct)*2 < len(sentence_to_compare):
				logger.trace("Length of sentence to compare is twice as big - break.")
				break
		logger.trace("Best end {}, best ratio {}.", best_end_index, best_ratio)
		return best_end_index		
		
	def get_sentences(self, words, repaired_text):
		sentences = []
		
		if len(words) == 0:
			logger.debug("There are 0 words in sentence {}.", repaired_text)
			return sentences
		
		# split repaired_text in sentences
		nlp = German()
		nlp.add_pipe('sentencizer')
		doc = nlp(repaired_text)
		repaired_text_sentences = [str(sent).strip() for sent in doc.sents]
		
		word_index = 0
		for rts in repaired_text_sentences:	
			words_in_sentence = [token.text for token in nlp(rts)]
			end_index = self.best_sentence_fit(words, word_index, words_in_sentence)
			confidences = [x.confidence for x in words[word_index:end_index]]
			avg_conf = 0
			min_conf = 0
			if len(confidences):
				avg_conf = sum(confidences) / (len(confidences) + 0.00000001)
				min_conf = min(confidences)
			
			logger.debug("Word count {}, start_index {}, end_index {}".format(len(words), word_index, end_index))
			
			if end_index < word_index:
				logger.warning("Sentence {} in text {} is messed up. Skipping entire audio file.", rts, repaired_text)
				return sentences
				
			sentences.append(AnnotatedSequence(rts, words[word_index].start, words[end_index].end, None, avg_conf, min_conf))
			word_index = end_index + 1
		return sentences
		
	def fix_audio(self, audio_file, tmp_dir, sample_rate):
		file_name, file_extension = os.path.splitext(os.path.basename(audio_file))
		audio = None
		is_video = False
		if file_extension.lower() == '.mp3':
			audio = AudioSegment.from_mp3(audio_file)
		elif file_extension.lower() == '.wav':
			audio = AudioSegment.from_wav(audio_file)
		elif file_extension.lower() == '.m4a':
			audio = AudioSegment.from_file(audio_file)
		elif file_extension.lower() == '.mp4':
			audio = AudioSegment.from_file(audio_file)
			is_video = True
		else:
			logger.error("File format {} not supported.", file_extension)
			sys.exit(0)
			
		audio = audio.set_channels(1)
		audio = audio.set_frame_rate(sample_rate)
		audio = audio.set_sample_width(2)
		file_name = os.path.join(tmp_dir, file_name + ".wav")
		file_exists = os.path.exists(file_name)
		audio.export(file_name, format='wav', bitrate="64k")
		return file_name, is_video, audio.duration_seconds, file_exists
		
def get_files(dir_or_mp3_file):
	files = []
	if os.path.exists(dir_or_mp3_file):
		if os.path.isfile(dir_or_mp3_file):
			files.append(dir_or_mp3_file)
		elif os.path.isdir(dir_or_mp3_file):
			if not dir_or_mp3_file.endswith("/"):
				dir_or_mp3_file += "/"
			search_path = dir_or_mp3_file + "**/*.mp3"
			for file in glob.glob(search_path, recursive=True):
				files.append(file)
	return files
	
def create_transcript_batch(files, batch_index, start_index, output_dir, tmp_dir, wav_dir, min_audio_length, max_audio_length, min_sentence_length, sample_rate):
	t = Transcriber()
	counter = start_index
	metadata_file = os.path.join(output_dir, "metadata{}.csv".format(batch_index))
	total_seconds = 0
	with open(metadata_file, 'w', encoding="utf-8") as csvfile:
		for file_index, f in enumerate(tqdm(files, desc="Processing files in batch {}".format(batch_index), position=batch_index)):
			logger.debug("Batch {}: Processing file {}/{} (audio so far: {})...", batch_index, file_index, len(files), timedelta(seconds=total_seconds))
							
			fixed_audio_file, is_video, audio_duration, file_existed = t.fix_audio(f, tmp_dir, sample_rate)
			
			if file_existed:
				total_seconds +=audio_duration				
				logger.debug("File {} already processed. skipping...", fixed_audio_file)
				continue	
			
			annotated_words, full_text = t.get_words_from_text(fixed_audio_file)
			repaired_text = t.repair_text(full_text)
			sentences = t.get_sentences(annotated_words, repaired_text)
			logger.debug("Found {} sentences in {}.", len(sentences), f)
			
			timestamp = datetime.now().microsecond
			audio = AudioSegment.from_wav(fixed_audio_file)
			
			total_seconds += audio.duration_seconds
					
			for anse in sentences:
				#timestamp = datetime.now().microsecond
				extract = audio[int(anse.start*1000):int(anse.end*1000)]
				if extract.duration_seconds >= min_audio_length:
					if extract.duration_seconds <= max_audio_length:
						if len(anse.sequence) >= min_sentence_length:
							extract.export(os.path.join(wav_dir, str(counter) + ".wav"), format='wav')
							
							csvfile.write(str(counter) + "|" + anse.sequence + "\n")
							
							with open(os.path.join(wav_dir, str(counter) + ".txt"), 'w', encoding="utf-8") as txtfile:
								txtfile.write(anse.sequence)
							counter +=1
						else:
							logger.debug("Sentence {} is not long enough.", anse.sequence)
					else:
						logger.debug("Audio of sentence {} is too long with {} seconds.", anse.sequence, extract.duration_seconds)
				else:
					logger.debug("Audio of sentence {} is too short with only {} seconds.", anse.sequence, extract.duration_seconds)
	logger.info("Batch {} finished. Waiting for others...", batch_index)
	
def create_transcript(dir_or_mp3_file, out_dir, min_sentence_length, min_audio_length, max_audio_length, number_of_processes, sample_rate):
	logger.info("Creating transcript...")
	start = timer()
	
	max_processes = multiprocessing.cpu_count()
	logger.info("Your max. number of parallel processes is {}.", max_processes)
	if number_of_processes > max_processes:
		logger.error("You don't have enough cpus to start {} processes. Maximum number of processes is {}. Exiting.", number_of_processes, max_processes)
		return
	
	files = get_files(dir_or_mp3_file)
	logger.info("There are {} mp3 files in {}. Creating {} batches...", len(files), dir_or_mp3_file, number_of_processes)
	
	# create folders
	timestamp = datetime.now().microsecond
	output_dir = "out" + str(timestamp)
	tmp_dir = os.path.join(output_dir, "tmp")
	
	if out_dir:
		output_dir = out_dir
		tmp_dir = os.path.join(output_dir, "tmp")
		
	wav_dir = os.path.join(output_dir, 'wavs')
	
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
		
	if not os.path.exists(wav_dir):
		os.mkdir(wav_dir)
		
	if not os.path.exists(tmp_dir):
		os.mkdir(tmp_dir)
		
	logger.debug("Output directory is {}, wav directory is {}, temp directory is {}.", output_dir, wav_dir, tmp_dir)
		
	# split batches
	processes = []
	batches = np.array_split(files, number_of_processes)
	start_index = 0
	for i, batch in enumerate(batches):
		p = Process(target=create_transcript_batch, args=(batch, i, start_index, output_dir, tmp_dir, wav_dir, min_audio_length, max_audio_length, min_sentence_length, sample_rate))
		start_index += len(batch)
		p.start()
		processes.append(p)
		
	# wait for all processes to finish
	for p in processes:
		p.join()

	# Todo: Join metadata files

	filenames = [os.path.join(output_dir, "metadata{}.csv".format(i)) for i in range(number_of_processes)]
	metadata_file = os.path.join(output_dir, "metadata.csv")
	with open(metadata_file, 'w') as outfile:
		for names in filenames:
			with open(names) as infile:
				outfile.write(infile.read())
			outfile.write("\n")
		
	end = timer()
	logger.info("Done in {}. Total audio time: {} \n".format(timedelta(seconds=end-start), 1))# timedelta(seconds=total_seconds)))
			
if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--dir_or_mp3_file", dest="dir_or_mp3_file", help="A directory containing multiple mp3 files or a single mp3 file.", type=str)
	parser.add_argument("-o", "--out_dir", dest="out_dir", help="Directory to store the generated dataset in.", type=str)
	parser.add_argument("-s", "--min_sentence_length", dest="min_sentence_length", help="Minimum of characters in a sentence.", type=int, default=5)
	parser.add_argument("-m", "--min_audio_length", dest="min_audio_length", help="Minimum length of the generated audio in seconds.", type=int, default=2)
	parser.add_argument("-l", "--max_audio_length", dest="max_audio_length", help="Maximum length of the generated audio in seconds.", type=int, default=30)
	parser.add_argument("-p", "--number_of_processes", dest="number_of_processes", help="The number of processes to process the data.", type=int, default=4)
	parser.add_argument("-r", "--sample_rate", dest="sample_rate", help="The sample rate of the input data.", type=int, default=22050)
	
	args = parser.parse_args()
	logger.remove()
	logger.add(sys.stderr, level="INFO")
	create_transcript(args.dir_or_mp3_file, args.out_dir, args.min_sentence_length, args.min_audio_length, args.max_audio_length, args.number_of_processes, args.sample_rate)
