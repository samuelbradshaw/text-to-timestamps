# Third-party libraries
import librosa
import torch
import torchaudio
from transformers import pipeline, AutoModel, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoProcessor, Wav2Vec2ForCTC
from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available

# Internal imports
import text_to_timestamps.utils


def transcribe_transformers(method, audio_path, model_name, lang = 'en', lang_fb = 'eng', word_timestamps = True, device = 'cuda', compute_type_torch = torch.float16, batch_size = 16):

  result = { 'text': '', 'words': [], 'data': {}, }
  
  attn = 'sdpa'
  if is_flash_attn_3_available():
    attn = 'flash_attention_3'
  elif is_flash_attn_2_available():
    attn = 'flash_attention_2'
  
  # CrisperWhisper (Transformers)
  if method == 'crisperwhisper-transformers':
    model = AutoModelForSpeechSeq2Seq.from_pretrained(f'nyrahealth/{model_name}', torch_dtype = compute_type_torch, use_safetensors = True)
    processor = AutoProcessor.from_pretrained(f'nyrahealth/{model_name}')
    
    pipe = pipeline(
      'automatic-speech-recognition',
      model = model,
      tokenizer = processor.tokenizer,
      feature_extractor = processor.feature_extractor,
      torch_dtype = compute_type_torch,
      device = device,
      framework = 'pt',
      model_kwargs = {'attn_implementation': attn},
      chunk_length_s = 30,
      batch_size = batch_size,
      ignore_warning = True,
      generate_kwargs = {'language': 'en', 'task': 'transcribe'},
      return_timestamps = 'word' if word_timestamps else False,
    )
    
    result['data'] = pipe(audio_path)
    result['text'] = result['data']['text'].strip()
    if word_timestamps:
      for word in result['data']['chunks']:
        result['words'].append({ 'text': word['text'].strip(), 'start': word['timestamp'][0], 'end': word['timestamp'][1] })
  
  
  # Distil-Whisper (Transformers)
  elif method == 'distil-whisper-transformers':
    model = AutoModelForSpeechSeq2Seq.from_pretrained(f'distil-whisper/{model_name}', torch_dtype = compute_type_torch, use_safetensors = True).to(device)
    processor = AutoProcessor.from_pretrained(f'distil-whisper/{model_name}')
    
    pipe = pipeline(
      'automatic-speech-recognition',
      model = model,
      tokenizer = processor.tokenizer,
      feature_extractor = processor.feature_extractor,
      torch_dtype = compute_type_torch,
      device = device,
      framework = 'pt',
      model_kwargs = {'attn_implementation': attn},
      chunk_length_s = 30,
      batch_size = batch_size,
      ignore_warning = True,
      generate_kwargs = {'language': 'en', 'task': 'transcribe'},
    )
    
    result['data'] = pipe(audio_path)
    result['text'] = result['data']['text'].strip()
  
  
  # Granite Speech (Transformers)
  elif method == 'granite-speech-transformers':
    speech_granite_processor = AutoProcessor.from_pretrained(f'ibm-granite/{model_name}')
    tokenizer = speech_granite_processor.tokenizer
    speech_granite = AutoModelForSpeechSeq2Seq.from_pretrained(f'ibm-granite/{model_name}').to(device)
    
    wav, sr = torchaudio.load(audio_path, normalize = True)
    wav = torchaudio.functional.resample(wav, orig_freq = sr, new_freq = 16000)
    wav = torch.mean(wav, dim = 0).unsqueeze(0)
    
    chat = [
      {
        'role': 'system',
        'content': 'Knowledge Cutoff Date: April 2024.\nToday\'s Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant',
      },
      {
        'role': 'user',
        'content': '<|audio|>can you transcribe the speech into a written format?',
      }
    ]
    
    text = tokenizer.apply_chat_template(chat, tokenize = False, add_generation_prompt = True)
    
    model_inputs = speech_granite_processor(
      text,
      wav,
      device = device,
      return_tensors = 'pt',
    ).to(device)
     
    model_outputs = speech_granite.generate(
      **model_inputs,
      max_new_tokens = 200,
      num_beams = 4,
      do_sample = False,
      min_length = 1,
      top_p = 1.0,
      repetition_penalty = 1.0,
      length_penalty = 1.0,
      temperature = 1.0,
      bos_token_id = tokenizer.bos_token_id,
      eos_token_id = tokenizer.eos_token_id,
      pad_token_id = tokenizer.pad_token_id,
    )
    
    # Transformers includes the input IDs in the response.
    num_input_tokens = model_inputs['input_ids'].shape[-1]
    new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim = 0)
    
    result['data'] = tokenizer.batch_decode(new_tokens, add_special_tokens = False, skip_special_tokens = True)
    for segment in result['data']:
      result['text'] += segment + ' '
    result['text'] = result['text'].strip()
      
  
  # LiteASR (Transformers)
  elif method == 'liteasr-transformers':
    
    model = AutoModel.from_pretrained(f'efficient-speech/lite-whisper-{model_name}', trust_remote_code = True)
    model.to(compute_type_torch).to(device)
    
    processor = AutoProcessor.from_pretrained(f'openai/whisper-{model_name}')
    audio, _ = librosa.load(audio_path, sr = 16000)
    
    input_features = processor(audio, sampling_rate = 16000, return_tensors = 'pt').input_features
    input_features = input_features.to(compute_type_torch).to(device)
    
    predicted_ids = model.generate(input_features)
    result['data'] = processor.batch_decode(predicted_ids, skip_special_tokens = True)
    for segment in result['data']:
      result['text'] += segment + ' '
    result['text'] = result['text'].strip()


  # MMS (Transformers)
  elif method == 'mms-transformers':
    ignore_mismatched_sizes = False
    if lang_fb != 'eng':
      ignore_mismatched_sizes = True
    
    model = Wav2Vec2ForCTC.from_pretrained(f'facebook/{model_name}', target_lang = lang_fb, ignore_mismatched_sizes = ignore_mismatched_sizes, torch_dtype = compute_type_torch, use_safetensors = True)
    processor = AutoProcessor.from_pretrained(f'facebook/{model_name}', target_lang = lang_fb)
    
    pipe = pipeline(
      'automatic-speech-recognition',
      model = model,
      tokenizer = processor.tokenizer,
      feature_extractor = processor.feature_extractor,
      torch_dtype = compute_type_torch,
      device = device,
      framework = 'pt',
      model_kwargs = {'attn_implementation': attn, 'target_lang': lang_fb, 'ignore_mismatched_sizes': ignore_mismatched_sizes},
      chunk_length_s = 30,
      batch_size = batch_size,
      ignore_warning = True,
      generate_kwargs = {'language': lang_fb, 'task': 'transcribe'},
    )
    
    result['data'] = pipe(audio_path)
    result['text'] = result['data']['text'].strip()
  
  
  # SeamlessM4T (Transformers)
  elif method == 'seamlessm4t-transformers':
    model = AutoModelForSpeechSeq2Seq.from_pretrained(f'facebook/{model_name}', torch_dtype = compute_type_torch, use_safetensors = True)
    processor = AutoProcessor.from_pretrained(f'facebook/{model_name}')
    
    pipe = pipeline(
      'automatic-speech-recognition',
      model = model,
      tokenizer = processor.tokenizer,
      feature_extractor = processor.feature_extractor,
      torch_dtype = compute_type_torch,
      device = device,
      framework = 'pt',
      model_kwargs = {'attn_implementation': attn},
      chunk_length_s = 30,
      batch_size = batch_size,
      ignore_warning = True,
      generate_kwargs = {'tgt_lang': lang_fb},
    )
    
    result['data'] = pipe(audio_path)
  
  
  # Whisper (Transformers)
  elif method == 'whisper-transformers':
    model = AutoModelForSpeechSeq2Seq.from_pretrained(f'openai/whisper-{model_name}', torch_dtype = compute_type_torch, use_safetensors = True)
    processor = AutoProcessor.from_pretrained(f'openai/whisper-{model_name}')
    
    pipe = pipeline(
      'automatic-speech-recognition',
      model = model,
      tokenizer = processor.tokenizer,
      feature_extractor = processor.feature_extractor,
      torch_dtype = compute_type_torch,
      device = device,
      framework = 'pt',
      model_kwargs = {'attn_implementation': attn},
      chunk_length_s = 30,
      batch_size = batch_size,
      ignore_warning = True,
      generate_kwargs = {'language': 'en', 'task': 'transcribe'},
      return_timestamps = 'word' if word_timestamps else False,
    )
    
    result['data'] = pipe(audio_path)
    result['text'] = result['data']['text'].strip()
    if word_timestamps:
      for word in result['data']['chunks']:
        result['words'].append({ 'text': word['text'].strip(), 'start': word['timestamp'][0], 'end': word['timestamp'][1] })
  
  return result