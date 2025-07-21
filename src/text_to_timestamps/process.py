# Python standard libraries
import os
import sys
import csv

# Internal imports
from text_to_timestamps.prepare import prepare
from text_to_timestamps.transcribe import transcribe
from text_to_timestamps.align import get_words, sort_words
from text_to_timestamps.output import output_to_file
import text_to_timestamps.utils


# Process a transcription/alignment job
def process(job_id, lang, audio, transcript = None, config = text_to_timestamps.utils.default_config, **config_args):
  if not (job_id and lang and audio):
    text_to_timestamps.utils.write(f'Error: One or more required inputs are missing (required inputs: job_id, lang, audio)\n', styles = ['red'], is_error = True)
    return sys.exit(1)
  
  job_id = str(job_id)
  config = text_to_timestamps.utils.parse_config(config, **config_args)
  
  audio_path = prepare(job_id, audio, config['general']['workfiles_directory'], voice_isolation_method = config['general']['voice_isolation_method'], force = config['general']['overwrite_previous_workfiles'])
  
  # Get transcript
  words = None
  if transcript:
    if os.path.isfile(transcript):
      with open(transcript, 'r', encoding = 'utf-8') as f:
        transcript = f.read()
    transcript = transcript.replace('\\n', '\n')
  elif config['transcribe']['method']:
    transcription_result = transcribe(config['transcribe']['method'], audio_path, lang = lang, get_words_if_available = (config['align']['method'] is None), model_size = config['transcribe']['model_size'], model_name = config['transcribe']['model_name'], batch_size = config['transcribe']['batch_size'])
    transcript = transcription_result['text']
    words = transcription_result['words'] or None
  else:
    text_to_timestamps.utils.write(f'Error: A transcript or transcription method must be provided\n', styles = ['red'], is_error = True)
    return sys.exit(1)
    
  # Align transcript words to audio
  if not words or (config['align']['method'] and config['align']['method'] != config['transcribe']['method']):
    words = get_words(config['align']['method'], audio_path, transcript, lang = lang, model_size = config['align']['model_size'], model_name = config['align']['model_name'], batch_size = config['align']['batch_size'])
  sorted_words = sort_words(words, transcript, output_config = config['output'])
  
  # Output to file
  output_to_file(job_id, sorted_words, config['general']['output_directory'], output_config = config['output'], additional_metadata = config['additional_metadata'], force = config['general']['overwrite_previous_output'])
  
  # Return aligned transcript
  timing_data = {
    'job_id': job_id,
    'text': transcript,
    'words': sorted_words,
  }
  return timing_data


# Process a batch of transcription/alignment jobs from a CSV or TSV file
def process_batch(input_csv, config = None, **config_args):
  timing_data_list = []
  delimiter = '\t' if os.path.splitext(input_csv)[-1] == '.tsv' else ','
  with open(input_csv, 'r', encoding = 'utf-8') as f:
    rows = list(csv.DictReader(f, delimiter = delimiter))
    for rw, row in enumerate(rows):
      job_id = row.get('job_id')
      lang = row.get('lang')
      audio = row.get('audio')
      transcript = row.get('transcript')
      
      for key, value in row.items():
        if '.' in key and key not in config_args:
          config_args[key] = value
      
      text_to_timestamps.utils.write(f'Processing: {job_id} ({rw + 1} of {len(rows)})\n')
      timing_data = process(job_id, lang, audio, transcript = transcript, config = config, **config_args)
      timing_data_list.append(timing_data)
  
  return timing_data_list

