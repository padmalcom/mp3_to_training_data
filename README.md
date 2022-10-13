# mp3-to-training-data

This tool extracts sentences and split audio from mp3, mp3 and wav files so that they can be used as training data for speech-to-text and text-to-speech applications,
e.g. https://github.com/padmalcom/Real-Time-Voice-Cloning-German

<a href="https://www.buymeacoffee.com/padmalcom" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

## Installation
- Download vosk models from https://alphacephei.com/vosk/models:
	- STT model: https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip
	- Puctuation and case restauration: https://alphacephei.com/vosk/models/vosk-recasepunc-de-0.21.zip
- Extract both into the application directory.
- Download BERT model from https://huggingface.co/dbmdz/bert-base-german-uncased
	- Put all 8 files in application directory/dbmdz/bert_base_german-uncased
- Create a conda environment and install all requirements from the requirements.txt. Install torch according to your cuda setup.

## Run
- Run using python main.py --dir_or_mp3_file "E:\my_audio_directory\" or python main.py --dir_or_mp3_file "E:\my_audio_file.mp3"

## Output
- The tool creates an output folder, called out[timestamp]. This folder contains:
	- a metadata.csv with filenames (excluding extensions) and texts
	- a wavs folder containing:
		- audio split into sentences and a corresponding txt file for each wav file with its spoken content

## Limitations
- This tool has only been tested with german text.
- All other languages require other models or even other implementations of stt and gramatical restauration.

