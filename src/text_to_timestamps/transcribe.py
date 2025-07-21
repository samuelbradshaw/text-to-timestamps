# Python standard libraries
import os
import io

# Third-party libraries
import torch
import librosa
import soundfile as sf

# Internal imports
from text_to_timestamps.transcribe_kyutai import transcribe_kyutai
from text_to_timestamps.transcribe_transformers import transcribe_transformers
import text_to_timestamps.utils


# Transcribe an audio file
def transcribe(method, audio_path, lang = 'en', get_words_if_available = False, model_size = 'small', model_name = None, batch_size = 16):
  if method not in text_to_timestamps.utils.supported_transcribe_methods:
    text_to_timestamps.utils.write(f'Error: Transcription method "{method}" is not supported\n', styles = ['red'], is_error = True)
    return sys.exit(1)
  
  method_name = text_to_timestamps.utils.supported_transcribe_methods[method]
  if not method_name:
    return {}
  
  text_to_timestamps.utils.write(f'Transcribing with {method_name}\n\n')
  
  device = text_to_timestamps.utils.get_best_available_device(method)
  compute_type = 'int8' if device == 'cpu' else 'float16'
  model_name = model_name or text_to_timestamps.utils.get_model_name(method, model_size)
  
  # TODO: Map BCP 47 to all of Facebook's supported language codes (for MMS and SeamlessM4T)
  map_bcp47_to_fb_lang = { 'en': 'eng', }
  lang_fb = map_bcp47_to_fb_lang.get(lang, 'en')
  compute_type_torch = getattr(torch, compute_type)
  
  result = {
    'text': '',
    'words': [],
    'data': {},
  }
  
  # distil-whisper (MLX)
  if method == 'distil-whisper-mlx':
    import mlx_whisper
    result['data'] = mlx_whisper.transcribe(audio_path, path_or_hf_repo = f'mlx-community/{model_name}', word_timestamps = get_words_if_available, language = lang)
    result['text'] = result['data']['text'].strip()
    for segment in result['data']['segments']:
      for word in segment['words']:
        result['words'].append({ 'text': word['word'].strip(), 'start': float(word['start']), 'end': float(word['end']) })
  
  # EasyMMS
  if method == 'easymms':
    pass
#     # TODO: Lots of dependency version errors
#     import easymms.models.asr
#     model_path = text_to_timestamps.utils.hf_hub_from_cache(f'facebook/{model_name}', 'pytorch_model.bin')
#     asr = easymms.models.asr.ASRModel(model = model_path)
#     result['data'] = asr.transcribe([audio_path], lang = lang_fb, align = False)
  
  # faster-whisper
  elif method == 'faster-whisper':
    import faster_whisper
    model = faster_whisper.WhisperModel(model_name, device = device, compute_type = compute_type)
    if batch_size > 1:
      model = faster_whisper.BatchedInferencePipeline(model = model)
    result['data'] = model.transcribe(audio_path, word_timestamps = get_words_if_available, batch_size = batch_size, language = lang)
    for segment in result['data'][0]:
      result['text'] += segment.text + ' '
      for word in segment.words:
        result['words'].append({ 'text': word.word.strip(), 'start': float(word.start), 'end': float(word.end) })
    result['text'] = result['text'].strip()
  
  # ForceAlign
  if method == 'forcealign':
    from forcealign import speech_to_text
    result['data'] = speech_to_text(audio_path)
    result['text'] = result['data'].strip()

  # Lhotse
  if method == 'lhotse':
    from lhotse import Recording, RecordingSet, annotate_with_whisper
    # Convert to single-track audio
    audio_data, sr = librosa.load(audio_path, mono = True)
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sr, format = 'WAV')
    buffer.seek(0)
    recording = Recording.from_bytes(buffer.read(), audio_path)
    recordings = RecordingSet.from_recordings([recording])
    result['data'] = annotate_with_whisper(recordings, language = lang, model_name = model_name, device = device)
    for segment in next(result['data']).supervisions:
      result['text'] += segment.text + ' '
    result['text'] = result['text'].strip()
  
  # PaddleSpeech
  elif method == 'paddlespeech':
    pass
#     # TODO: Lots of dependency version errors
#     from paddlespeech.cli.asr.infer import ASRExecutor
#     asr = ASRExecutor()
#     audio, sr = librosa.load(audio_path, sr = 16000, mono = True)
#     result['data'] = asr(audio_file = audio_path, lang = lang, sample_rate = sr, model = model_name)    
  
  # Parakeet MLX
  elif method == 'parakeet-mlx':
    import parakeet_mlx
    model = parakeet_mlx.from_pretrained(f'mlx-community/{model_name}')
    result['data'] = model.transcribe(audio_path, chunk_duration = 60 * 2.0, overlap_duration = 15.0)
    result['text'] = result['data'].text.strip()
    for sentence in result['data'].sentences:
      for token in sentence.tokens:
        result['words'].append({ 'text': token.text.strip(), 'start': float(token.start), 'end': float(token.end) })
  
  # pywhispercpp
  elif method == 'pywhispercpp':
    import pywhispercpp.model
    model = pywhispercpp.model.Model(model_name, token_timestamps = True, max_len = 1, split_on_word = True)
    result['data'] = model.transcribe(audio_path, language = lang)
    for word in result['data']:
      result['text'] += word.text + ' '
      result['words'].append({ 'text': word.text.strip(), 'start': float(word.t0), 'end': float(word.t1) })
    result['text'] = result['text'].strip()
  
  # stable-ts
  elif method == 'stable-ts':
    import stable_whisper
    model = stable_whisper.load_model(model_name, device = device)
    result['data'] = model.transcribe(audio_path, language = lang, task = 'transcribe').to_dict()
    result['text'] = result['data']['text'].strip()
    for segment in result['data']['segments']:
      for word in segment['words']:
        result['words'].append({ 'text': word['word'].strip(), 'start': float(word['start']), 'end': float(word['end']) })
  
  # stable-ts (faster-whisper)
  elif method == 'stable-ts-faster-whisper':
    import stable_whisper
    model = stable_whisper.load_faster_whisper(model_name, device = device)
    result['data'] = model.transcribe(audio_path, language = lang, task = 'transcribe').to_dict()
    result['text'] = result['data']['text'].strip()
    for segment in result['data']['segments']:
      for word in segment['words']:
        result['words'].append({ 'text': word['word'].strip(), 'start': float(word['start']), 'end': float(word['end']) })
  
  # stable-ts (MLX Whisper)
  elif method == 'stable-ts-mlx-whisper':
    import stable_whisper
    model = stable_whisper.load_mlx_whisper(model_name, device = device)
    result['data'] = model.transcribe(audio_path, language = lang, task = 'transcribe').to_dict()
    result['text'] = result['data']['text'].strip()
    for segment in result['data']['segments']:
      for word in segment['words']:
        result['words'].append({ 'text': word['word'].strip(), 'start': float(word['start']), 'end': float(word['end']) })
  
  # Whisper JAX
  elif method == 'whisper-jax':
    from whisper_jax import FlaxWhisperPipline
    import jax.numpy as jnp
    pipeline = FlaxWhisperPipline(f'openai/whisper-{model_name}', dtype = jnp.bfloat16, batch_size = batch_size)
    result['data'] = pipeline(audio_path, task = 'transcribe', return_timestamps = True)
    result['text'] = result['data']['text'].strip()
  
  # Whisper (MLX)
  elif method == 'whisper-mlx':
    import mlx_whisper
    result['data'] = mlx_whisper.transcribe(audio_path, path_or_hf_repo = f'mlx-community/{model_name}', word_timestamps = get_words_if_available, language = lang)
    result['text'] = result['data']['text'].strip()
    for segment in result['data']['segments']:
      for word in segment['words']:
        result['words'].append({ 'text': word['word'].strip(), 'start': float(word['start']), 'end': float(word['end']) })
  
  # whisper-mps
  elif method == 'whisper-mps':
    from whisper_mps import whisper as whisper_mps
    result['data'] = whisper_mps.transcribe(audio_path, model = model_name, language = lang)
    result['text'] = result['data']['text'].strip()
  
  # Whisper (OpenAI)
  elif method == 'whisper-openai':
    import whisper
    model = whisper.load_model(model_name, device = device)
    result['data'] = model.transcribe(audio_path, word_timestamps = get_words_if_available, fp16 = device != 'cpu', language = lang)
    result['text'] = result['data']['text'].strip()
    for segment in result['data']['segments']:
      for word in segment['words']:
        result['words'].append({ 'text': word['word'].strip(), 'start': float(word['start']), 'end': float(word['end']) })
  
  # whisper-timestamped
  elif method == 'whisper-timestamped':
    import whisper_timestamped
    audio = whisper_timestamped.load_audio(audio_path)
    model = whisper_timestamped.load_model(model_name, device = device)
    result['data'] = model.transcribe(audio_path, word_timestamps = get_words_if_available, fp16 = device != 'cpu', language = lang)
    result['text'] = result['data']['text'].strip()
    for segment in result['data']['segments']:
      for word in segment['words']:
        result['words'].append({ 'text': word['word'].strip(), 'start': float(word['start']), 'end': float(word['end']) })
  
  # WhisperS2T
  elif method == 'whispers2t':
    import whisper_s2t
    model = whisper_s2t.load_model(model_identifier = model_name, backend = 'CTranslate2', compute_type = compute_type, device = device, asr_options = {'word_timestamps': get_words_if_available})
    result['data'] = model.transcribe_with_vad([audio_path], lang_codes = [lang], tasks = ['transcribe'], batch_size = batch_size)
    result['text'] = result['data'][0][0]['text'].strip()
    for word in result['data'][0][0]['word_timestamps']:
      result['words'].append({ 'text': word['word'].strip(), 'start': float(word['start']), 'end': float(word['end']) })
  
  # WhisperX
  elif method == 'whisperx':
    import whisperx
    model = whisperx.load_model(model_name, device, compute_type = compute_type, language = lang, task = 'transcribe')
    audio = whisperx.load_audio(audio_path)
    result['data'] = model.transcribe(audio, batch_size = batch_size)
    for segment in result['data']['segments']:
      result['text'] += segment['text'] + ' '
    result['text'] = result['text'].strip()
  
  # Transformers
  elif method in ['crisperwhisper-transformers', 'distil-whisper-transformers', 'granite-speech-transformers', 'liteasr-transformers', 'mms-transformers', 'seamlessm4t-transformers', 'whisper-transformers']:
    result = transcribe_transformers(method, audio_path, model_name, lang = lang, lang_fb = lang_fb, word_timestamps = get_words_if_available, device = device, compute_type_torch = compute_type_torch, batch_size = batch_size)
  
  # Kyutai
  elif method in ['kyutai-stt', 'kyutai-stt-mlx']:
    if method == 'kyutai-stt':
      result['data'] = transcribe_kyutai(audio_path, device = device, use_mlx = False)
    elif method == 'kyutai-stt-mlx':
      result['data'] = transcribe_kyutai(audio_path, device = device, use_mlx = True)
    result['text'] = result['data']['text'].strip()
    for word in result['data']['words']:
      result['words'].append({ 'text': word['text'].strip(), 'start': float(word['start']), 'end': float(word['end']) })
  
  return result
