# Python standard libraries
import math
import json

# Third-party libraries (kyutai)
import moshi.models
import rustymimi
import moshi_mlx.models
import moshi_mlx.utils

# Third-party libraries (other)
import numpy as np
import librosa
import torch
import sentencepiece
import mlx.core
import mlx.nn
import tqdm

# Internal imports
import text_to_timestamps.utils


def transcribe_kyutai(audio_path, device = 'cuda', use_mlx = False):
  if use_mlx and device != 'mps':
    use_mlx = False
  
  hf_repo = 'kyutai/stt-2.6b-en-mlx' if use_mlx else 'kyutai/stt-2.6b-en'
  
  # Load config data
  lm_config_path = text_to_timestamps.utils.hf_hub_from_cache(hf_repo, 'config.json')
  with open(lm_config_path, 'r') as f:
    lm_config = json.load(f)
  mimi_weights_path = text_to_timestamps.utils.hf_hub_from_cache(hf_repo, lm_config['mimi_name'])
  moshi_weights_path = text_to_timestamps.utils.hf_hub_from_cache(hf_repo, 'model.safetensors')
  text_tokenizer_path = text_to_timestamps.utils.hf_hub_from_cache(hf_repo, lm_config['tokenizer_name'])
  stt_config = lm_config.get('stt_config', {})
  
  # Define constants
  PADDING_TOKEN_ID = lm_config.get('existing_text_padding_id', 3)
  BOUNDARY_TOKEN_ID = lm_config.get('existing_text_end_padding_id', 0)
  SAMPLE_RATE = 24000
  FRAME_SIZE = 1920
  FRAME_RATE = SAMPLE_RATE / FRAME_SIZE

  # Load audio
  audio, sr = librosa.load(audio_path, sr = SAMPLE_RATE, mono = True)
  audio_duration = librosa.get_duration(y = audio, sr = sr)
  audio = np.array([audio])

  # Add padding at the beginning and end of the audio
  audio_silence_prefix_seconds = stt_config.get('audio_silence_prefix_seconds', 1.0)
  audio_delay_seconds = stt_config.get('audio_delay_seconds', 2.5)
  pad_left = math.ceil(audio_silence_prefix_seconds) * SAMPLE_RATE
  pad_right = math.ceil(audio_delay_seconds) * SAMPLE_RATE
  if audio.shape[-1] % FRAME_SIZE != 0:
    # Add more padding at the end if needed, to make length a multiple of the frame size
    pad_right += FRAME_SIZE - audio.shape[-1] % FRAME_SIZE
  audio = np.pad(audio, pad_width=[(0, 0), (pad_left, pad_right)], mode='constant')

  text_tokens = []
  text_tokenizer = sentencepiece.SentencePieceProcessor(text_tokenizer_path)  
  num_steps = np.shape(audio)[-1] // FRAME_SIZE
  
  if use_mlx:
    lm_config = moshi_mlx.models.LmConfig.from_config_dict(lm_config)
    
    # Load the audio tokenizer
    num_codebooks = max(lm_config.generated_codebooks, lm_config.other_codebooks)
    audio_tokenizer = rustymimi.Tokenizer(mimi_weights_path, num_codebooks = num_codebooks)
    
    # Load Lm
    lm_model = moshi_mlx.models.Lm(lm_config)
    lm_model.set_dtype(mlx.core.bfloat16)
    lm_model.load_weights(moshi_weights_path, strict = True)
    ct = None
    if lm_model.condition_provider is not None:
      ct = lm_model.condition_provider.condition_tensor('description', 'very_good')
    lm_model.warmup(ct)

    # Load LmGen
    lm_gen = moshi_mlx.models.LmGen(
      model = lm_model, max_steps = num_steps,
      text_sampler = moshi_mlx.utils.Sampler(top_k = 25, temp = 0),
      audio_sampler = moshi_mlx.utils.Sampler(top_k = 250, temp = 0.8),
    )
    
    # Run inference
    for idx in tqdm.tqdm(range(num_steps)):
      pcm_data = audio[:, idx * FRAME_SIZE:(idx + 1) * FRAME_SIZE]
      audio_tokens = audio_tokenizer.encode_step(pcm_data[None, 0:1])
      audio_tokens = mlx.core.array(audio_tokens).transpose(0, 2, 1)[:, :, :lm_config.other_codebooks]
      text_token = lm_gen.step(audio_tokens[0], ct)
      text_tokens.append(text_token[0].item())

  else:
    # Load audio tokenizer and models
    checkpoint_info = moshi.models.loaders.CheckpointInfo.from_hf_repo(hf_repo)
    audio_tokenizer = checkpoint_info.get_mimi(device = device)
    lm_model = checkpoint_info.get_moshi(device = device, dtype = torch.bfloat16)
    lm_gen = moshi.models.LMGen(lm_model, temp = 0, temp_text = 0.0)

    # Run inference
    audio = torch.from_numpy(audio).to(device)
    audio_chunks = torch.split(audio[:, None], FRAME_SIZE, dim = -1)
    with audio_tokenizer.streaming(1), lm_gen.streaming(1):
      for audio_chunk in tqdm.tqdm(audio_chunks):
        audio_tokens = audio_tokenizer.encode(audio_chunk)
        text_token = lm_gen.step(audio_tokens)
        if text_token is not None:
          text_tokens.append(text_token)
    text_tokens = torch.concat(text_tokens, dim = -1).flatten().tolist()
  
  # Decode tokens and get timestamps
  words = []
  offset_seconds = audio_silence_prefix_seconds + audio_delay_seconds
  for idx in range(num_steps):
    text_token = text_tokens[idx]
    
    # Word boundary
    if text_token in (BOUNDARY_TOKEN_ID,):
      seconds = ((idx + 1) / FRAME_RATE) - offset_seconds
      if words:
        words[-1]['end'] = seconds
      words.append({'text': '', 'start': seconds, 'end': audio_duration,})
    
    # Word
    elif text_token not in (BOUNDARY_TOKEN_ID, PADDING_TOKEN_ID,):
      words[-1]['text'] += text_tokenizer.id_to_piece(text_token).replace('▁', ' ')
  
  result = {
    'text': ''.join([word['text'] for word in words]),
    'words': words,
  }
  return result
