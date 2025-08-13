# -------- VOICE ISOLATION --------

supported_voice_isolation_methods = {
  'audio-separator': {
    'name': 'Audio Separator',
    'url': 'https://github.com/nomadkaraoke/python-audio-separator',
    'license': 'MIT',
    'pythonPackage': 'audio_separator',
  },
  'deepfilternet': {
    'name': 'DeepFilterNet',
    'url': 'https://github.com/Rikorose/DeepFilterNet',
    'license': 'MIT',
    'pythonPackage': 'df',
  },
  'demucs': {
    'name': 'Demucs',
    'url': 'https://github.com/adefossez/demucs',
    'license': 'MIT',
    'pythonPackage': 'demucs',
  },
  'noisereduce': {
    'name': 'noisereduce',
    'url': 'https://github.com/timsainb/noisereduce',
    'license': 'MIT',
    'pythonPackage': 'noisereduce',
  },
  'open-unmix': {
    'name': 'Open-Unmix',
    'url': 'https://github.com/sigsep/open-unmix-pytorch',
    'license': 'MIT',
    'pythonPackage': 'openunmix',
  },
  
  # Not implemented due to conflicting dependencies or other complications
#   'spleeter': {
#     'name': 'Spleeter',
#     'url': 'https://github.com/deezer/spleeter,
#     'license': 'MIT',
#     'pythonPackage': 'spleeter',
#   }
}


# -------- TRANSCRIPTION --------

supported_transcribe_methods = {
  'distil-whisper-mlx': {
    'name': 'distil-whisper (MLX)',
    'url': 'https://huggingface.co/mlx-community/distil-whisper-large-v3',
    'license': 'MIT',
    'pythonPackage': 'mlx_whisper',
  },
  'faster-whisper': {
    'name': 'faster-whisper',
    'url': 'https://github.com/SYSTRAN/faster-whisper',
    'license': 'MIT',
    'pythonPackage': 'faster_whisper',
  },
  'forcealign': {
    'name': 'ForceAlign',
    'url': 'https://github.com/lukerbs/forcealign',
    'license': 'MIT',
    'pythonPackage': 'forcealign',
  },
  'lhotse': {
    'name': 'Lhotse',
    'url': 'https://github.com/lhotse-speech/lhotse',
    'license': 'Apache-2.0',
    'pythonPackage': 'lhotse',
  },
  'parakeet-mlx': {
    'name': 'Parakeet MLX',
    'url': 'https://github.com/senstella/parakeet-mlx',
    'license': 'Apache-2.0',
    'pythonPackage': 'parakeet_mlx',
  },
  'pywhispercpp': {
    'name': 'pywhispercpp',
    'url': 'https://github.com/absadiki/pywhispercpp',
    'license': 'MIT',
    'pythonPackage': 'pywhispercpp',
  },
  'stable-ts': {
    'name': 'stable-ts',
    'url': 'https://github.com/jianfch/stable-ts',
    'license': 'MIT',
    'pythonPackage': 'stable-ts',
  },
  'stable-ts-faster-whisper': {
    'name': 'stable-ts (faster-whisper)',
    'url': 'https://github.com/jianfch/stable-ts',
    'license': 'MIT',
    'pythonPackage': 'stable-ts',
  },
  'stable-ts-mlx-whisper': {
    'name': 'stable-ts (MLX Whisper)',
    'url': 'https://github.com/jianfch/stable-ts',
    'license': 'MIT',
    'pythonPackage': 'stable-ts',
  },
  'whisper-jax': {
    'name': 'Whisper JAX',
    'url': 'https://github.com/JnJarvis/whisper-jax',
    'license': 'Apache-2.0',
    'pythonPackage': 'whisper_jax',
  },
  'whisper-mlx': {
    'name': 'Whisper (MLX)',
    'url': 'https://github.com/ml-explore/mlx-examples/tree/main/whisper',
    'license': 'MIT',
    'pythonPackage': 'mlx_whisper',
  },
  'whisper-mps': {
    'name': 'whisper-mps',
    'url': 'https://github.com/AtomGradient/whisper-mps',
    'license': 'MIT',
    'pythonPackage': 'whisper_mps',
  },
  'whisper-openai': {
    'name': 'Whisper (OpenAI)',
    'url': 'https://github.com/openai/whisper',
    'license': 'MIT',
    'pythonPackage': 'openai-whisper',
  },
  'whisper-timestamped': {
    'name': 'whisper-timestamped',
    'url': 'https://github.com/linto-ai/whisper-timestamped',
    'license': 'AGPL-3.0',
    'pythonPackage': 'whisper_timestamped',
  },
  'whispers2t': {
    'name': 'WhisperS2T',
    'url': 'https://github.com/shashikg/WhisperS2T',
    'license': 'MIT',
    'pythonPackage': 'whisper-s2t',
  },
  'whisperx': {
    'name': 'WhisperX',
    'url': 'https://github.com/m-bain/whisperX',
    'license': 'BSD-2-Clause',
    'pythonPackage': 'whisperx',
  },
  # Tranformers
  'crisperwhisper-transformers': {
    'name': 'CrisperWhisper (Transformers)',
    'url': 'https://github.com/nyrahealth/CrisperWhisper',
    'license': 'BSD-2-Clause',
    'pythonPackage': 'transformers',
  },
  'distil-whisper-transformers': {
    'name': 'Distil-Whisper (Transformers)',
    'url': 'https://github.com/huggingface/distil-whisper',
    'license': 'MIT',
    'pythonPackage': 'transformers',
  },
  'granite-speech-transformers': {
    'name': 'Granite Speech (Transformers)',
    'url': 'https://github.com/ibm-granite/granite-speech-models',
    'license': 'Apache-2.0',
    'pythonPackage': 'transformers',
  },
  'liteasr-transformers': {
    'name': 'LiteASR (Transformers)',
    'url': 'https://github.com/efeslab/LiteASR',
    'license': 'Apache-2.0',
    'pythonPackage': 'transformers',
  },
  'mms-transformers': {
    'name': 'MMS (Transformers)',
    'url': 'https://github.com/facebookresearch/fairseq/tree/main/examples/mms',
    'license': 'MIT',
    'pythonPackage': 'transformers',
  },
  'seamlessm4t-transformers': {
    'name': 'SeamlessM4T (Transformers)',
    'url': 'https://github.com/facebookresearch/seamless_communication',
    'license': 'CC BY-NC 4.0',
    'pythonPackage': 'transformers',
  },
  'whisper-transformers': {
    'name': 'Whisper (Transformers)',
    'url': 'https://huggingface.co/docs/transformers/en/model_doc/whisper',
    'license': 'Apache-2.0',
    'pythonPackage': 'transformers',
  },
  # Kyutai
  'kyutai-stt': {
    'name': 'Kyutai STT',
    'url': 'https://github.com/kyutai-labs/delayed-streams-modeling',
    'license': 'CC-BY 4.0',
    'pythonPackage': 'moshi',
  },
  'kyutai-stt-mlx': {
    'name': 'Kyutai STT (MLX)',
    'url': 'https://github.com/kyutai-labs/delayed-streams-modeling',
    'license': 'CC-BY 4.0',
    'pythonPackage': 'moshi-mlx',
  },
  
  # Not implemented due to conflicting dependencies or other complications
#   'easymms': {
#     'name': 'EasyMMS',
#     'url': 'https://github.com/absadiki/easymms',
#     'license': 'CC BY-NC-ND 4.0',
#     'pythonPackage': 'easymms',
#   },
#   'paddlespeech': {
#     'name': 'PaddleSpeech',
#     'url': 'https://github.com/PaddlePaddle/PaddleSpeech',
#     'license': 'Apache-2.0',
#     'pythonPackage': 'paddlespeech',
#   },
}


# -------- FORCED ALIGNMENT --------

supported_align_methods = {
  'forcealign-align': {
    'name': 'ForceAlign',
    'url': 'https://github.com/lukerbs/forcealign',
    'license': 'MIT',
    'pythonPackage': 'forcealign',
  },
  'stable-ts-align': {
    'name': 'stable-ts',
    'url': 'https://github.com/jianfch/stable-ts',
    'license': 'MIT',
    'pythonPackage': 'stable_whisper',
  },
  'stable-ts-faster-whisper-align': {
    'name': 'stable-ts (faster-whisper)',
    'url': 'https://github.com/jianfch/stable-ts',
    'license': 'MIT',
    'pythonPackage': 'stable_whisper',
  },
  'stable-ts-mlx-whisper-align': {
    'name': 'stable-ts (MLX Whisper)',
    'url': 'https://github.com/jianfch/stable-ts',
    'license': 'MIT',
    'pythonPackage': 'stable_whisper',
  },
  'whisperx-align': {
    'name': 'WhisperX',
    'url': 'https://github.com/m-bain/whisperX',
    'license': 'BSD-2-Clause',
    'pythonPackage': 'whisperx',
  },
  
  # Not implemented due to conflicting dependencies or other complications
#   'aeneas-align': {
#     'name': 'aeneas',
#     'url': 'https://github.com/akki2825/aeneas',
#     'license': 'AGPL-3.0',
#     'pythonPackage': 'aeneas',
#   },
#   'afaligner-align': {
#     'name': 'afaligner',
#     'url': 'https://github.com/r4victor/afaligner',
#     'license': 'MIT',
#     'pythonPackage': 'afaligner',
#   },
#   'ctc-forced-aligner-align': {
#     'name': 'ctc-forced-aligner',
#     'url': 'https://github.com/MahmoudAshraf97/ctc-forced-aligner',
#     'license': 'BSD',
#     'pythonPackage': 'ctc_forced_aligner',
#   },
#   'easymms-align': {
#     'name': 'EasyMMS',
#     'url': 'https://github.com/absadiki/easymms',
#     'license': 'CC BY-NC-ND 4.0',
#     'pythonPackage': 'easymms',
#   },
#   'lhotse-align': {
#     'name': 'Lhotse',
#     'url': 'https://github.com/lhotse-speech/lhotse',
#     'license': 'Apache-2.0',
#     'pythonPackage': 'lhotse',
#   },
#   'montreal-forced-aligner-align': {
#     'name': 'Montreal Forced Aligner',
#     'url': 'https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner',
#     'license': 'MIT',
#     'pythonPackage': 'montreal_forced_aligner',
#   },
#   'timething-align': {
#     'name': 'Timething',
#     'url': 'https://github.com/feldberlin/timething',
#     'license': 'MIT',
#     'pythonPackage': 'timething',
#   },
}


# ------------------------------------------

# Python standard libraries
import os
import sys
import json
import copy
from importlib.metadata import version

# Third-party libraries
import numpy as np
from huggingface_hub import hf_hub_download, try_to_load_from_cache
import torch
import librosa

default_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'default_config.json')
with open(default_config_path, 'r', encoding = 'utf-8') as f:
  default_config = json.load(f)

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
          if value.lower().strip() in ['none', 'null', 'nil', 'na', 'n/a', '']:
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
            translation_table = str.maketrans(';', ',', '[\'"]')
            value = value.translate(translation_table).lower()
            value = [val.strip() for val in value.split(',')]
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


def get_method_info(method):
  info = supported_voice_isolation_methods.get(method) or supported_transcribe_methods.get(method) or supported_align_methods.get(method)
  info['pythonPackageVersion'] = version(info['pythonPackage'])
  return info