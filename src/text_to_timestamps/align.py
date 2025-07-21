# Python standard libraries
import os
import io
import tempfile
import re
from collections import defaultdict
import copy

# Third-party libraries
import librosa
import soundfile as sf

# Internal imports
import text_to_timestamps.utils


# Get timestamps for each word in a transcript
def get_words(method, audio_path, transcript, lang = 'en', model_size = 'small', model_name = None, batch_size = 16):
  if method not in text_to_timestamps.utils.supported_align_methods:
    text_to_timestamps.utils.write(f'Error: Alignment method "{method}" is not supported\n', styles = ['red'], is_error = True)
    return sys.exit(1)
  
  method_name = text_to_timestamps.utils.supported_align_methods[method]
  if not method_name:
    return {}
  
  text_to_timestamps.utils.write(f'Aligning with {method_name}\n\n')

  device = text_to_timestamps.utils.get_best_available_device(method)
  compute_type = 'int8' if device == 'cpu' else 'float16'
  model_name = model_name or text_to_timestamps.utils.get_model_name(method, model_size)
  audio_duration = librosa.get_duration(path = audio_path)
  
  words = []
  
  # Clean up transcript for easier alignment
  cleaned_transcript = transcript
  cleaned_transcript = cleaned_transcript.replace(r'—', '— ') # Em dashes
  cleaned_transcript = re.sub(r'^\d+\.?\s', '', cleaned_transcript, flags = re.MULTILINE) # Leading numbers
  cleaned_transcript = re.sub(r'<[^>]+?>', ' ', cleaned_transcript) # HTML tags
  
  # aeneas
  if method == 'aeneas-align':
    pass
    # TODO: Not implemented
#     from aeneas.task import Task
#     from aeneas.executetask import ExecuteTask
#     task = Task(config_string = 'task_language=eng|is_text_type=plain|os_task_file_format=json')
#     with tempfile.NamedTemporaryFile(mode = 'w+t', delete = False, encoding = 'utf-8', suffix = '.txt') as temp_file:
#       temp_file.write(cleaned_transcript)
#       transcript_path = temp_file.name
#     task.audio_file_path_absolute = audio_path
#     task.text_file_path_absolute = transcript_path
#     ExecuteTask(task).execute()
#     data = task.sync_map.json_string
  
  # afaligner
  if method == 'afaligner':
    pass
    # TODO: Not implemented
#     import afaligner.align

  # ctc-forced-aligner
  if method == 'ctc-forced-aligner':
    pass
    # TODO: Not implemented

  # ForceAlign
  if method == 'forcealign-align':
    import nltk
    nltk_data_path = os.path.join(os.path.expanduser("~"), '.cache', 'nltk_data')
    nltk.data.path.append(nltk_data_path)
    nltk.download('cmudict', download_dir = nltk_data_path)
    nltk.download('averaged_perceptron_tagger', download_dir = nltk_data_path)
    nltk.download('averaged_perceptron_tagger_eng', download_dir = nltk_data_path)
    from forcealign import ForceAlign
    align = ForceAlign(audio_file = audio_path, transcript = cleaned_transcript)
    data = align.inference()
    for word in data:
      words.append({ 'text': word.word.strip(), 'start': float(word.time_start), 'end': float(word.time_end) })

  # Lhotse
  if method == 'lhotse-align':
    pass
    # TODO: Not implemented
#     from lhotse import Recording, RecordingSet, align_with_torchaudio
#     # Convert to single-track audio
#     audio_data, sr = librosa.load(audio_path, mono = True)
#     buffer = io.BytesIO()
#     sf.write(buffer, audio_data, sr, format = 'WAV')
#     buffer.seek(0)
#     recording = Recording.from_bytes(buffer.read(), audio_path)
#     recordings = RecordingSet.from_recordings([recording])
#     data = align_with_torchaudio(recordings, device = device)
#     print(list(data))

  # Montreal Forced Aligner
  if method == 'montreal-forced-aligner-align':
    pass
    # TODO: Not implemented

  # stable-ts
  if method == 'stable-ts-align':
    import stable_whisper
    model = stable_whisper.load_model(model_name, device = device)
    data = model.align(audio_path, cleaned_transcript, language = lang, original_split = True, vad = False)
    for segment in data.to_dict()['segments']:
      for word in segment['words']:
        words.append({ 'text': word['word'].strip(), 'start': float(word['start']), 'end': float(word['end']) })

  # stable-ts (faster-whisper)
  if method == 'stable-ts-faster-whisper-align':
    import stable_whisper
    model = stable_whisper.load_faster_whisper(model_name, device = device)
    data = model.align(audio_path, cleaned_transcript, language = lang, original_split = True, vad = False)
    for segment in data.to_dict()['segments']:
      for word in segment['words']:
        words.append({ 'text': word['word'].strip(), 'start': float(word['start']), 'end': float(word['end']) })

  # stable-ts (MLX Whisper)
  if method == 'stable-ts-mlx-whisper-align':
    import stable_whisper
    model = stable_whisper.load_mlx_whisper(model_name, device = device)
    data = model.align(audio_path, cleaned_transcript, language = lang, original_split = True, vad = False)
    for segment in data.to_dict()['segments']:
      for word in segment['words']:
        words.append({ 'text': word['word'].strip(), 'start': float(word['start']), 'end': float(word['end']) })

  # Timething
  if method == 'timething-align':
    pass
    # TODO: Not implemented
    # https://github.com/feldberlin/timething/issues/32
    # https://github.com/feldberlin/timething/issues/33
#     from timething.cli import align_long
#     with tempfile.NamedTemporaryFile(mode = 'w+t', delete = False, encoding = 'utf-8', suffix = '.txt') as temp_file:
#       temp_file.write(cleaned_transcript)
#       transcript_path = temp_file.name
#     data = align_long(audio_file = audio_path, transcript_file = transcript_path, language = lang, batch_size = batch_size, use_gpu = device != 'cpu')

  # WhisperX
  if method == 'whisperx-align':
    import whisperx
    model, metadata = whisperx.load_align_model(language_code = 'en', device = device)
    data = whisperx.align([{'text': cleaned_transcript, 'start': 0, 'end': audio_duration}], model, metadata, audio_path, device)
    for word in data['word_segments']:
      words.append({ 'text': word['word'].strip(), 'start': float(word['start']), 'end': float(word['end']) })
  
  return words


# Align timestamped words to transcript more exactly
def sort_words(unsorted_words, transcript, output_config = {}):
  initial_output_config = copy.deepcopy(text_to_timestamps.utils.default_config['output'])
  initial_output_config.update(output_config)
  output_config = initial_output_config
  
  words = []
  words_by_block = defaultdict(list)
  words_by_phrase = defaultdict(list)
  
  block_counter = 0
  phrase_counter = 0
  word_counter = 0
  character_counter = 0
  phrase_character_count = 0
  for word in unsorted_words:
    if not word.get('text'):
      continue
    
    # Handle whitespace and other untimed text before the current word
    previous_word = words[-1] if words else {}
    untimed_text = ''
    prefix = ''
    if not transcript.startswith(word['text']):
      while transcript and not transcript.startswith(word['text']):
        untimed_text += transcript[0]
        phrase_character_count += 1
        transcript = transcript[1:]
      if previous_word:
        if '\n' in untimed_text:
          previous_word_suffix, line_break, prefix = untimed_text.rpartition('\n')
          previous_word['suffix'] = previous_word_suffix + line_break
          character_counter += len(previous_word['suffix'])
        else:
          previous_word['suffix'] = untimed_text
          character_counter += len(untimed_text)
      else:
        prefix = untimed_text
    
    num_leading_newlines = len(untimed_text.split('\n')) - 1
    transcript = transcript.removeprefix(word['text'])
    phrase_character_count += len(word['text'])
    
    # New block
    if num_leading_newlines > 1:
      block_counter += 1
      phrase_counter += 1
      phrase_character_count = 0
    
    # New phrase
    elif previous_word and (
      ('single_line_breaks' in output_config['break_phrases_at'] and num_leading_newlines == 1)
      or ('sentence_punctuation' in output_config['break_phrases_at'] and any([p in previous_word['text'] for p in '.;:–—…!?']))
      or ('character_count' in output_config['break_phrases_at'] and phrase_character_count > output_config['max_phrase_character_count'])
    ):
      phrase_counter += 1
      phrase_character_count = 0
    
    text = word['text']
    start_offset = character_counter + len(prefix)
    end_offset = character_counter + len(prefix) + len(text)
    word_info = {
      'text': text,
      'prefix': prefix,
      'suffix': '',
      'startTime': max(0, word.get('start', previous_word.get('start') or 0) + output_config['time_offset']),
      'endTime': max(0, word.get('end', previous_word.get('start') or 0) + output_config['time_offset']),
      'startOffset': start_offset,
      'endOffset': end_offset,
      'wordNumber': word_counter + 1,
      'phraseNumber': phrase_counter + 1,
      'blockNumber': block_counter + 1,
    }
    words.append(word_info)    
    words_by_phrase[phrase_counter + 1].append(word_info)
    words_by_block[block_counter + 1].append(word_info)
    word_counter += 1
    character_counter = end_offset

  return {
    'words': words,
    'wordsByBlock': dict(words_by_block),
    'wordsByPhrase': dict(words_by_phrase),
  }