# Third-party libraries
import os
import sys
import json
import copy

# Third-party libraries
import numpy as np
from huggingface_hub import hf_hub_download, try_to_load_from_cache
import torch
import librosa


default_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'default_config.json')
with open(default_config_path, 'r', encoding = 'utf-8') as f:
  default_config = json.load(f)

supported_voice_isolation_methods = {
  'audio-separator': 'Audio Separator', # https://github.com/nomadkaraoke/python-audio-separator
  'deepfilternet': 'DeepFilterNet', # https://github.com/Rikorose/DeepFilterNet
  'demucs': 'Demucs', # https://github.com/adefossez/demucs
  'noisereduce': 'noisereduce', # https://github.com/timsainb/noisereduce
  'open-unmix': 'Open-Unmix', # https://github.com/sigsep/open-unmix-pytorch
  
  # Not implemented due to conflicting dependencies or other complications
#   'spleeter': 'Spleeter', # https://github.com/deezer/spleeter
}

supported_transcribe_methods = {
  'distil-whisper-mlx': 'distil-whisper (MLX)', # https://huggingface.co/mlx-community/distil-whisper-large-v3
  'faster-whisper': 'faster-whisper', # https://github.com/SYSTRAN/faster-whisper
  'forcealign': 'ForceAlign', # https://github.com/lukerbs/forcealign
  'parakeet-mlx': 'Parakeet MLX', # https://github.com/senstella/parakeet-mlx
  'pywhispercpp': 'pywhispercpp', # https://github.com/absadiki/pywhispercpp
  'stable-ts': 'stable-ts', # https://github.com/jianfch/stable-ts
  'stable-ts-faster-whisper': 'stable-ts (faster-whisper)', # https://github.com/jianfch/stable-ts
  'stable-ts-mlx-whisper': 'stable-ts (MLX Whisper)', # https://github.com/jianfch/stable-ts
  'whisper-jax': 'Whisper JAX', # https://github.com/sanchit-gandhi/whisper-jax
  'whisper-mlx': 'Whisper (MLX)', # https://github.com/ml-explore/mlx-examples/tree/main/whisper
  'whisper-mps': 'whisper-mps', # https://github.com/AtomGradient/whisper-mps
  'whisper-openai': 'Whisper (OpenAI)', # https://github.com/openai/whisper
  'whisper-timestamped': 'whisper-timestamped', # https://github.com/linto-ai/whisper-timestamped
  'whispers2t': 'WhisperS2T', # https://github.com/shashikg/WhisperS2T
  'whisperx': 'WhisperX', # https://github.com/m-bain/whisperX
  # Tranformers
  'crisperwhisper-transformers': 'CrisperWhisper (Transformers)', # https://github.com/nyrahealth/CrisperWhisper
  'distil-whisper-transformers': 'Distil-Whisper (Transformers)', # https://github.com/huggingface/distil-whisper
  'granite-speech-transformers': 'Granite Speech (Transformers)', # https://github.com/ibm-granite/granite-speech-models
  'liteasr-transformers': 'LiteASR (Transformers)', # https://github.com/efeslab/LiteASR
  'lhotse': 'Lhotse', # https://github.com/lhotse-speech/lhotse
  'mms-transformers': 'MMS (Transformers)', # https://github.com/facebookresearch/fairseq/tree/main/examples/mms
  'seamlessm4t-transformers': 'SeamlessM4T (Transformers)', # https://github.com/facebookresearch/seamless_communication
  'whisper-transformers': 'Whisper (Transformers)', # https://huggingface.co/docs/transformers/en/model_doc/whisper
  # Kyutai
  'kyutai-stt': 'Kyutai STT', # https://github.com/kyutai-labs/delayed-streams-modeling
  'kyutai-stt-mlx': 'Kyutai STT (MLX)', # https://github.com/kyutai-labs/delayed-streams-modeling
  
  # Not implemented due to conflicting dependencies or other complications
#   'easymms': 'EasyMMS', # https://github.com/absadiki/easymms
#   'paddlespeech': 'PaddleSpeech', # https://github.com/PaddlePaddle/PaddleSpeech
}

supported_align_methods = {
  'forcealign-align': 'ForceAlign', # https://github.com/lukerbs/forcealign
  'stable-ts-align': 'stable-ts', # https://github.com/jianfch/stable-ts
  'stable-ts-faster-whisper-align': 'stable-ts (faster-whisper)', # https://github.com/jianfch/stable-ts
  'stable-ts-mlx-whisper-align': 'stable-ts (MLX Whisper)', # https://github.com/jianfch/stable-ts
  'whisperx-align': 'WhisperX', # https://github.com/m-bain/whisperX
  
  # Not implemented due to conflicting dependencies or other complications
#   'aeneas-align': 'aeneas', # https://github.com/akki2825/aeneas
#   'afaligner-align': 'afaligner', # https://github.com/r4victor/afaligner
#   'ctc-forced-aligner-align': 'ctc-forced-aligner', # https://github.com/MahmoudAshraf97/ctc-forced-aligner
#   'easymms-align': 'EasyMMS', # https://github.com/absadiki/easymms
#   'lhotse-align': 'Lhotse', # https://github.com/lhotse-speech/lhotse
#   'montreal-forced-aligner-align': 'Montreal Forced Aligner', # https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
#   'timething-align': 'Timething', # https://github.com/feldberlin/timething
}

def get_model_name(method, model_size):
  model_map = {
    'crisperwhisper-transformers': {
      'large': 'CrisperWhisper',
    },
    'distil-whisper-transformers': {
      'large': 'distil-large-v3.5',
    },
    'distil-whisper-mlx': {
      'large': 'distil-whisper-large-v3',
    },
    'easymms': {
      'large': 'mms-1b-all',
    },
    'granite-speech-transformers': {
      'medium': 'granite-speech-3.3-2b',
      'large': 'granite-speech-3.3-8b',
    },
    'kyutai-stt': {
      'large': 'stt-2.6b-en',
    },
    'kyutai-stt-mlx': {
      'large': 'stt-2.6b-en-mlx',
    },
    'mms-transformers': {
      'large': 'mms-1b-all',
    },
    'paddlespeech': {
      'large': 'deepspeech2offline_librispeech',
    },
    'parakeet-mlx': {
      'small': 'parakeet-tdt_ctc-110m',
      'large': 'parakeet-tdt_ctc-1.1b',
    },
    'seamlessm4t-transformers': {
      'large': 'seamless-m4t-v2-large',
    },
    'whisper-mlx': {
      'small': 'whisper-small-mlx',
      'medium': 'whisper-medium-mlx',
      'large': 'whisper-large-v2-mlx',
    },
    'whisper-openai': {
      'small': 'small',
      'medium': 'medium',
      'large': 'large-v2',
    },
  }
  
  model_source = method
  if method in ['faster-whisper', 'liteasr-transformers', 'lhotse', 'pywhispercpp', 'stable-ts', 'stable-ts-align', 'stable-ts-faster-whisper', 'stable-ts-faster-whisper-align', 'stable-ts-mlx-whisper', 'stable-ts-mlx-whisper-align', 'whisper-jax', 'whisper-mps', 'whisper-timestamped', 'whisper-transformers', 'whispers2t', 'whisperx', 'whisperx-align']:
    model_source = 'whisper-openai'
  
  model_name = model_map.get(model_source, {}).get(model_size)
  available_model_names = []
  if not model_name:
    for key, value in model_map.get(model_source, {}).items():
      available_model_names.append(value)
      if key in model_size:
        model_name = value
        break
  
  if not model_name:
    if available_model_names:
      model_name = available_model_names[0]
      write(f'Warning: Model not found for size "{model_size}". Falling back to "{model_name}".\n')
    else:
      write(f'Warning: Model not found for size "{model_size}".\n')
  
  return model_name


def get_best_available_device(method):
  device = 'cpu' # Any computer
  if torch.cuda.is_available():
    device = 'cuda' # Nvidia
  elif torch.backends.mps.is_available() and method not in [
    'faster-whisper', # https://github.com/SYSTRAN/faster-whisper/issues/911
    'lhotse', # https://github.com/openai/whisper/pull/382
    'stable-ts', # https://github.com/jianfch/stable-ts/issues/263
    'stable-ts-faster-whisper', # https://github.com/SYSTRAN/faster-whisper/issues/911
    'stable-ts-align', # https://github.com/jianfch/stable-ts/issues/263
    'stable-ts-faster-whisper-align', # https://github.com/SYSTRAN/faster-whisper/issues/911
    'whisper-openai', # https://github.com/openai/whisper/pull/382
    'whispers2t', # https://github.com/OpenNMT/CTranslate2/issues/1562
    'whisper-timestamped', # https://github.com/linto-ai/whisper-timestamped/issues/217
    'whisperx', # https://github.com/m-bain/whisperX/issues/109
  ]:
    device = 'mps' # Apple M-series
  
  if method in ['distil-whisper-mlx', 'kyutai-stt-mlx', 'parakeet-mlx', 'whisper-mlx', 'whisper-mps'] and device != 'mps':
    write(f'Warning: "{method}" not supported on device "{device}".\n')
    return None
  
  return device
  

# Build a config dict based on input and defaults
def parse_config(config, **config_args):
  # Config fallback order: config_args (from command line or CSV) –> custom config (passed in as a dict or file path –> default config
  
  # Get custom config
  custom_config = {}
  if not config:
    pass
  elif isinstance(config, dict):
    custom_config = config
  elif isinstance(config, str) and config[0] == '{':
    custom_config = json.loads(config)
  elif isinstance(config, str) and os.path.isfile(config):
    with open(config, 'r', encoding = 'utf-8') as f:
      custom_config = json.load(f)
  else:
    write(f'Warning: Invalid config object or path\n')
  
  # Update custom config from config_args
  if config_args:
    integer_fields = ['max_phrase_character_count', 'json_indent']
    float_fields = ['time_offset']
    boolean_fields = ['json_as_js', 'webvtt_timestamp_tags', 'include_text', 'include_prefixes', 'include_suffixes', 'include_start_times', 'include_end_times', 'include_start_offsets', 'include_end_offsets', 'include_word_numbers', 'include_phrase_numbers', 'include_block_numbers', 'overwrite_previous_workfiles', 'overwrite_previous_output']
    list_fields = ['format', 'granularity', 'break_phrases_at']
    for category_key, value in config_args.items():
      if '.' in category_key: # Example: { 'transcribe.method': 'whisper-mlx' }
        category, key = category_key.split('.')
        if category not in custom_config:
          custom_config[category] = {}
        
        # Convert string values to correct data types
        if isinstance(value, str):
          if value.lower().strip() in ['none', 'null', 'nil', 'na', 'n/a', ' ', '']:
            value = None
          elif key in integer_fields:
            value = int(value)
          elif key in float_fields:
            value = float(value)
          elif key in boolean_fields:
            if any(val in value.lower() for val in ['1', 't', 'y']):
              value = True
            elif any(val in value.lower() for val in ['0', 'f', 'n']):
              value = False
          elif key in list_fields:
            value = [val.strip() for val in value.lower().replace(';', ',').split(',')]
        
        custom_config[category][key] = value
  
  # Update default config with custom config values
  config = copy.deepcopy(default_config)
  if custom_config:
    for category, options in config.items():
      if category in custom_config:
        config[category].update(custom_config[category])
  
  return config


# Change the sample rate of a PCM audio stream (as a numpy array)
def resample(audio_np_array, original_SAMPLE_RATE, target_SAMPLE_RATE):
  if audio_np_array.ndim == 1: # Mono audio
    resampled = librosa.resample(audio_np_array, orig_sr = original_SAMPLE_RATE, target_sr = target_SAMPLE_RATE)
    return resampled.astype(np.float32)
  elif audio_np_array.ndim == 2: # Multi-channel audio
    channels, length = audio_np_array.shape
    resampled_channels = []
    for channel in range(channels):
      resampled_channel = librosa.resample(audio_np_array[channel, :], orig_sr = original_SAMPLE_RATE, target_sr = target_SAMPLE_RATE)
      resampled_channels.append(resampled_channel)
  return np.array(resampled_channels, dtype = np.float32)


def hf_hub_from_cache(repo_id, filename):
  cached_path = try_to_load_from_cache(repo_id, filename)
  return cached_path or hf_hub_download(repo_id, filename)


def write(text, styles = None, is_error = False):
  ansi_codes = {
    'red': '\033[91m',
    'bold': '\033[1m',
    'reset': '\033[0m',
  }
  
  styles_start = ''
  styles_end = ''
  if styles:
    for style in styles:
      styles_start += ansi_codes[style]
    styles_end = ansi_codes['reset']
  
  if is_error:
    sys.stderr.write(f'{styles_start}{text}{styles_end}')
  else:
    sys.stdout.write(f'{styles_start}{text}{styles_end}')


