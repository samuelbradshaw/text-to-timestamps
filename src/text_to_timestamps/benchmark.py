# Python standard libraries
import sys
import time

# Internal imports
import text_to_timestamps.utils


def benchmark(methods, audio_path, transcribe = True, align = True, transcript = None, model_size = None, device = None):
  
  model_size = model_size or 'small'
  device = device or text_to_timestamps.utils.get_best_available_device()
  batch_size = 1 # Not all methods support batching, so this levels the playing field
  
  for method in methods:
    utlis.load_method(method)
  
  run_benchmark('mlx-whisper', transcribe_mlx_whisper, audio_path,
    word_timestamps = word_timestamps, whisper_model = whisper_model, device = device)
#   run_benchmark('mlx-distil-whisper', transcribe_mlx_distil_whisper, audio_path,
#     word_timestamps = word_timestamps, device = device)
#   run_benchmark('Kyutai STT', transcribe_kyutai, audio_path,
#     device = device, use_mlx = False)
#   run_benchmark('Kyutai STT (MLX)', transcribe_kyutai, audio_path,
#     device = device, use_mlx = True)
#   run_benchmark('Whisper s2T Ctranslate', transcribe_whisper_s2t_ctranslate2, audio_path,
#     word_timestamps = word_timestamps, whisper_model = whisper_model, device = device, batch_size = batch_size)
#   run_benchmark('Whisper MPS', transcribe_whisper_mps, audio_path,
#     whisper_model = whisper_model)
#   run_benchmark('transcribe_whisper_timestamped', transcribe_whisper_timestamped, audio_path,
#     word_timestamps = word_timestamps, whisper_model = whisper_model, device = device)
#   run_benchmark('transcribe_faster_whisper', transcribe_faster_whisper, audio_path,
#     word_timestamps = word_timestamps, whisper_model = whisper_model, device = device)
#   run_benchmark('transcribe_whisper', transcribe_whisper, audio_path,
#     word_timestamps = word_timestamps, whisper_model = whisper_model, device = device)
#   run_benchmark('transcribe_hf_mms', transcribe_hf_mms, audio_path,
#     device = device, batch_size = batch_size)
#   run_benchmark('transcribe_hf_openai_whisper', transcribe_hf_openai_whisper, audio_path,
#     word_timestamps = word_timestamps, whisper_model = whisper_model, device = device, batch_size = batch_size)
#   run_benchmark('transcribe_hf_seamless', transcribe_hf_seamless, audio_path,
#     word_timestamps = word_timestamps, device = device, batch_size = batch_size)
#   run_benchmark('transcribe_hf_crisperwhisper', transcribe_hf_crisperwhisper, audio_path,
#     word_timestamps = word_timestamps, device = device, batch_size = batch_size)
#   run_benchmark('transcribe_hf_liteasr', transcribe_hf_liteasr, audio_path,
#     word_timestamps = word_timestamps, whisper_model = whisper_model, device = device, batch_size = batch_size)
#   run_benchmark('transcribe_hf_distil_whisper', transcribe_hf_distil_whisper, audio_path,
#     word_timestamps = word_timestamps, whisper_model = whisper_model, device = device, batch_size = batch_size)
#   run_benchmark('transcribe_hf_granite', transcribe_hf_granite, audio_path,
#     device = device)
#   run_benchmark('transcribe_parakeet_mlx', transcribe_parakeet_mlx, audio_path,
#     word_timestamps = False, device = device)
#   run_benchmark('transcribe_paddlepaddle_speech', transcribe_paddlepaddle_speech, audio_path,
#     device = device)
#   run_benchmark('transcribe_pywhispercpp', transcribe_pywhispercpp, audio_path,
#     word_timestamps = True, whisper_model = whisper_model, device = device)
#   run_benchmark('transcribe_whisper_jax', transcribe_whisper_jax, audio_path,
#     whisper_model = whisper_model, device = device, batch_size = batch_size)
  run_benchmark('transcribe_whisperx', transcribe_whisperx, audio_path,
    whisper_model = whisper_model, device = device, batch_size = batch_size)
  run_benchmark('transcribe_stable_ts', transcribe_stable_ts, audio_path,
    whisper_model = whisper_model, device = device, batch_size = batch_size)


def run_benchmark(name, function, audio_path, **kwargs):
  text_to_timestamps.utils.write.write(f'Transcribing with {name}\n', styles = ['bold'])
  t0 = time.perf_counter()
  result = transcribe(audio_path, **kwargs)
  run_time = time.perf_counter() - t0
  text_to_timestamps.utils.write.write(f'Time: {run_time}\n')
  text_to_timestamps.utils.write.write(f'Result: {result}\n')
  text_to_timestamps.utils.write.write('\n')
