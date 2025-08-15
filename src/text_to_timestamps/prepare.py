# Python standard libraries
import os
import sys
import shutil
import mimetypes

# Third-party libraries
import requests
import pydub
import librosa
import soundfile as sf
import torch
import torchaudio

# Internal imports
import text_to_timestamps.utils


# Prepare an audio file for transcription and/or alignment
def prepare(job_id, audio, workfiles_directory, voice_isolation_method = None, force = False):
  os.makedirs(workfiles_directory, exist_ok = True)
  
  wav_path = os.path.join(workfiles_directory, f'{job_id}.wav')
  text_to_timestamps.utils.write(f'Preparing: {job_id}\n')
  
  # Download or copy audio file to download directory
  if not os.path.isfile(wav_path) or force:
    if 'https://' in audio:
      text_to_timestamps.utils.write(f'  Downloading audio: {audio}\n')
      download_path = download(audio, workfiles_directory, job_id)
      extension = os.path.splitext(download_path)[-1]
    else:
      text_to_timestamps.utils.write(f'  Copying audio\n')
      extension = os.path.splitext(audio)[-1]
      download_path = os.path.join(workfiles_directory, f'{job_id}{extension}')
      shutil.copyfile(audio, download_path)
    
    # Convert to WAV if needed
    if download_path != wav_path:
      sound = pydub.AudioSegment.from_file(download_path)
      sound.export(wav_path, format = 'wav')
  
  clean_path = os.path.join(workfiles_directory, f'{job_id}.clean.wav')
  if voice_isolation_method and (not os.path.isfile(clean_path) or force):
    audio_path = clean(voice_isolation_method, job_id, wav_path, clean_path)
  else:
    audio_path = wav_path
  
  return audio_path
  

# Remove background noise or music
def clean(voice_isolation_method, job_id, wav_path, clean_path):
  text_to_timestamps.utils.write(f'  Cleaning audio with {voice_isolation_method}\n')
  
  device = text_to_timestamps.utils.get_best_available_device(voice_isolation_method)
  
  if voice_isolation_method == 'audio-separator':
    from audio_separator.separator import Separator
    separator = Separator(output_single_stem = 'Vocals', output_dir = os.path.dirname(clean_path))
    separator.load_model()
    output_files = separator.separate(wav_path, {'Vocals': job_id})
  
  elif voice_isolation_method == 'deepfilternet':
    from df.enhance import enhance, init_df, load_audio, save_audio
    audio_array, sr = load_audio(wav_path, sr = 48000)
    model, df_state = init_df()[:2]
    enhanced = enhance(model, df_state, audio_array)
    save_audio(clean_path, enhanced, df_state.sr())
  
  elif voice_isolation_method == 'demucs':
    import demucs.api
    separator = demucs.api.Separator(model = 'mdx_extra', device = device, jobs = 4, progress = True)
    original, separated = separator.separate_audio_file(wav_path)
    for stem, source in separated.items():
      if stem == 'vocals':
        demucs.api.save_audio(source, clean_path, samplerate = separator.samplerate)
  
  elif voice_isolation_method == 'noisereduce':
    import noisereduce
    audio_array, sr = librosa.load(wav_path)
    reduced_noise = noisereduce.reduce_noise(y = audio_array, sr = sr)
    sf.write(clean_path, reduced_noise, sr, format = 'WAV')
  
  elif voice_isolation_method == 'open-unmix':
    from openunmix.predict import separate
    waveform, sample_rate = torchaudio.load(wav_path)
    estimates = separate(waveform, rate = sample_rate, targets = ['vocals'], residual = True, device = device)
    torchaudio.save(clean_path, torch.squeeze(estimates['vocals']).to('cpu'), sample_rate)
  
  elif voice_isolation_method == 'spleeter':
    pass
#       from spleeter.separator import Separator
#       from spleeter.audio.adapter import AudioAdapter
#       separator = Separator('spleeter:2stems')
#       audio_loader = AudioAdapter.default()
#       waveform, sr = audio_loader.load(wav_path, sample_rate = 44100)
#       prediction = separator.separate(waveform)
#       sf.write(clean_path, prediction['vocals'], sr, format = 'WAV')
  
  return clean_path


# Download an audio file
def download(url, download_directory, job_id):
  os.makedirs(download_directory, exist_ok = True)
  try:
    headers = { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36', }
    with requests.get(url, headers = headers, stream = True) as r:
      extension = mimetypes.guess_extension(r.headers['Content-Type']) or '.mp3'
      download_path = os.path.join(download_directory, f'{job_id}{extension}')
      with open(download_path, 'wb') as bf:
        for chunk in r.iter_content(chunk_size = 10000):
          bf.write(chunk)
  except:
    text_to_timestamps.utils.write(f'Error: Failed to download: {url}\n', styles = ['red'], is_error = True)
    f'Error: Failed to download: {url}\n'
    return sys.exit(1)
  
  return download_path