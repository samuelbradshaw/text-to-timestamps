# Python standard libraries
import os
from datetime import datetime, date
import json
import csv
import copy

# Internal imports
import text_to_timestamps.utils


# Output timestamped words to a file
def output_to_file(job_id, sorted_words, output_directory, output_config = {}, additional_metadata = {}, force = True):
  os.makedirs(output_directory, exist_ok = True)
  
  initial_output_config = copy.deepcopy(text_to_timestamps.utils.default_config['output'])
  initial_output_config.update(output_config)
  output_config = initial_output_config
  
  today = date.today().isoformat()
  metadata = {
    '_': f'Generated {today} by Text to Timestamps (https://github.com/samuelbradshaw/text-to-timestamps)',
  }
  metadata.update(additional_metadata)
  json_metadata = json.dumps(metadata, ensure_ascii = False, indent = output_config['json_indent'] or 2)
  
  vtt = { 'block': '', 'phrase': '', 'word': '' }
  rows = { 'block': [], 'phrase': [], 'word': [] }
  
  if 'block' in output_config['granularity']:
    vtt['block'] += f'WEBVTT\n\nNOTE\n{json_metadata}'
    for sequence_number, words in sorted_words['wordsByBlock'].items():
      word_start_times = [word['startTime'] for word in words if word['startTime'] is not None]
      word_end_times = [word['endTime'] for word in words if word['endTime'] is not None]
      max_seconds = word_end_times[-1]
      if 'webvtt' in output_config['format']:
        vtt['block'] = vtt['block'].rstrip() + '\n\n'
        
        cue_metadata = {}
        if output_config['include_block_numbers']: cue_metadata['blockNumber'] = words[0]['blockNumber']
        vtt['block'] += (json.dumps(cue_metadata, ensure_ascii = False) if cue_metadata else str(sequence_number)) + '\n'
        
        vtt['block'] += format_timestamp(word_start_times[0], max_seconds) + ' --> ' + format_timestamp(word_end_times[-1], max_seconds) + '\n'
        for word in words:
          vtt['block'] += word['prefix']
          if output_config['webvtt_timestamp_tags'] and word['startTime'] is not None:
            vtt['block'] += '<' + format_timestamp(word['startTime'], max_seconds) + '>'
          vtt['block'] += word['text'] + word['suffix']
      if any([f in output_config['format'] for f in ['json', 'csv']]):
        row = {}
        if output_config['include_text']: row['text'] = ''.join([word['prefix'] + word['text'] + word['suffix'] for word in words])
        if output_config['include_start_times']: row['startTime'] = round(word_start_times[0], 3)
        if output_config['include_end_times']: row['endTime'] = round(word_end_times[-1], 3)
        if output_config['include_start_offsets']: row['startOffset'] = words[0]['startOffset'] - len(words[0]['prefix'])
        if output_config['include_end_offsets']: row['endOffset'] = words[-1]['endOffset'] + len(words[0]['suffix'])
        if output_config['include_block_numbers']: row['blockNumber'] = words[0]['blockNumber']
        rows['block'].append(row)
  
  if 'phrase' in output_config['granularity']:
    vtt['phrase'] += f'WEBVTT\n\nNOTE\n{json_metadata}'
    for sequence_number, words in sorted_words['wordsByPhrase'].items():
      word_start_times = [word['startTime'] for word in words if word['startTime'] is not None]
      word_end_times = [word['endTime'] for word in words if word['endTime'] is not None]
      max_seconds = word_end_times[-1]
      if 'webvtt' in output_config['format']:
        vtt['phrase'] = vtt['phrase'].rstrip() + '\n\n'
        
        cue_metadata = {}
        if output_config['include_phrase_numbers']: cue_metadata['phraseNumber'] = words[0]['phraseNumber']
        if output_config['include_block_numbers']: cue_metadata['blockNumber'] = words[0]['blockNumber']
        vtt['phrase'] += (json.dumps(cue_metadata, ensure_ascii = False) if cue_metadata else str(sequence_number)) + '\n'
        
        vtt['phrase'] += format_timestamp(word_start_times[0], max_seconds) + ' --> ' + format_timestamp(word_end_times[-1], max_seconds) + '\n'
        for word in words:
          vtt['phrase'] += word['prefix']
          if output_config['webvtt_timestamp_tags'] and word['startTime'] is not None:
            vtt['phrase'] += '<' + format_timestamp(word['startTime'], max_seconds) + '>'
          vtt['phrase'] += word['text'] + word['suffix']
        vtt['phrase'] = vtt['phrase'].rstrip() + '\n\n'
      if any([f in output_config['format'] for f in ['json', 'csv']]):
        row = {}
        if output_config['include_text']: row['text'] = ''.join([word['prefix'] + word['text'] + word['suffix'] for word in words])
        if output_config['include_start_times']: row['startTime'] = round(word_start_times[0], 3)
        if output_config['include_end_times']: row['endTime'] = round(word_end_times[-1], 3)
        if output_config['include_start_offsets']: row['startOffset'] = words[0]['startOffset'] - len(words[0]['prefix'])
        if output_config['include_end_offsets']: row['endOffset'] = words[-1]['endOffset'] + len(words[0]['suffix'])
        if output_config['include_phrase_numbers']: row['phraseNumber'] = words[0]['phraseNumber']
        if output_config['include_block_numbers']: row['blockNumber'] = words[0]['blockNumber']
        rows['phrase'].append(row)
  
  if 'word' in output_config['granularity']:
    vtt['word'] += f'WEBVTT\n\nNOTE\n{json_metadata}\n\n'
    max_seconds = next((d['endTime'] for d in reversed(sorted_words['words']) if d.get('endTime') is not None), None)
    for wd, word in enumerate(sorted_words['words']):
      sequence_number = wd + 1
      if 'webvtt' in output_config['format']:
        if word['startTime'] is not None:
          vtt['word'] = vtt['word'].rstrip() + '\n\n'
          
          cue_metadata = {}
          if output_config['include_word_numbers']: cue_metadata['wordNumber'] = word['wordNumber']
          if output_config['include_phrase_numbers']: cue_metadata['phraseNumber'] = word['phraseNumber']
          if output_config['include_block_numbers']: cue_metadata['blockNumber'] = word['blockNumber']
          vtt['word'] += (json.dumps(cue_metadata, ensure_ascii = False) if cue_metadata else str(sequence_number)) + '\n'
          
          vtt['word'] += format_timestamp(word['startTime'], max_seconds) + ' --> ' + format_timestamp(word['endTime'], max_seconds) + '\n'
        vtt['word'] += word['prefix']
        if output_config['webvtt_timestamp_tags'] and word['startTime'] is not None:
          vtt['word'] += '<' + format_timestamp(word['startTime'], max_seconds) + '>'
        vtt['word'] += word['text'] + word['suffix']
      if any([f in output_config['format'] for f in ['json', 'csv']]):
        row = {}
        if output_config['include_text']: row['text'] = word['text']
        if output_config['include_prefixes']: row['prefix'] = word['prefix']
        if output_config['include_suffixes']: row['suffix'] = word['suffix']
        if output_config['include_start_times']: row['startTime'] = round(word['startTime'], 3)
        if output_config['include_end_times']: row['endTime'] = round(word['endTime'], 3)
        if output_config['include_start_offsets']: row['startOffset'] = word['startOffset']
        if output_config['include_end_offsets']: row['endOffset'] = word['endOffset']
        if output_config['include_word_numbers']: row['wordNumber'] = word['wordNumber']
        if output_config['include_phrase_numbers']: row['phraseNumber'] = word['phraseNumber']
        if output_config['include_block_numbers']: row['blockNumber'] = word['blockNumber']
        rows['word'].append(row)
  
  for granularity_key in output_config['granularity']:
    if 'webvtt' in output_config['format']:
      output_path = os.path.join(output_directory, f'{job_id}.{granularity_key}.vtt')
      if not os.path.isfile(output_path) or force:
        with open(output_path, 'w') as f:
          f.write(vtt[granularity_key])
    if 'csv' in output_config['format']:
      keys = rows[granularity_key][0].keys()
      if output_config['csv_delimiter'] == '\t':
        output_path = os.path.join(output_directory, f'{job_id}.{granularity_key}.tsv')
      else:
        output_path = os.path.join(output_directory, f'{job_id}.{granularity_key}.csv')
      if not os.path.isfile(output_path) or force:
        with open(output_path, 'w', newline = '') as f:
          dict_writer = csv.DictWriter(f, keys, delimiter = output_config['csv_delimiter'])
          dict_writer.writeheader()
          dict_writer.writerows(rows[granularity_key])
    if 'json' in output_config['format']:
      content = {
        'metadata': metadata,
        'segments': rows[granularity_key],
      }
      if output_config['json_as_js']:
        output_path = os.path.join(output_directory, f'{job_id}.{granularity_key}.js')
        if not os.path.isfile(output_path) or force:
          with open(output_path, 'w') as f:
            json_string = json.dumps(content, ensure_ascii = False, indent = output_config['json_indent'], separators = (', ', ': ') if output_config['json_indent'] else (',', ':'))
            f.write(f'var timingData = timingData ?? new Object();\ntimingData["{job_id}"] = ' + json_string)
      else:
        output_path = os.path.join(output_directory, f'{job_id}.{granularity_key}.json')
        if not os.path.isfile(output_path) or force:
          with open(output_path, 'w') as f:
            json.dump(content, f, ensure_ascii = False, indent = output_config['json_indent'], separators = (', ', ': ') if output_config['json_indent'] else (',', ':'))


# Convert seconds to a WebVTT timestamp
def format_timestamp(seconds, max_seconds = None):
  dt = datetime.utcfromtimestamp(seconds)
  formatted_timestamp = dt.isoformat(sep = 'T', timespec = 'milliseconds').split('T')[-1]
  if max_seconds is not None and max_seconds < 3600:
    # Remove the hours place if the maximum duration is less than an hour
    formatted_timestamp = formatted_timestamp.split(':', 1)[-1]
  return formatted_timestamp

