# Python standard libraries
import argparse

# Internal imports
from text_to_timestamps.process import process, process_batch
import text_to_timestamps.utils

def main_cli():
  parser = argparse.ArgumentParser(description='Text to Timestamps')
  parser.add_argument('command', required = True, help='Command to run ("process" or "process_batch"). Required.')
  parser.add_argument('--job_id', help='Job ID. Required when running "process".')
  parser.add_argument('--lang', help='BCP 47 language tag. Required when running "process".')
  parser.add_argument('--audio', help='URL or path to audio file. Required when running "process".')
  parser.add_argument('--transcript', help='Path to transcript text file (or text string). Optional. Only applicable when running "process".')
  parser.add_argument('--csv_input', help='Path to input CSV. Required when running "process_batch".')
  parser.add_argument('--config', help='Path to config JSON file (or JSON string). Optional.')
  for category in text_to_timestamps.utils.default_config:
    for key, value in text_to_timestamps.utils.default_config[category].items():
      parser.add_argument(f'--{category}.{key}', help='Individual config value. Optional.')
  
  args = parser.parse_args()
  
  if args.command == 'process':
    result = process(
      args.job_id,
      args.lang,
      args.audio,
      transcript = args.transcript,
      config = args.config,
      **{ key: value for key, value in vars(args).items() if '.' in key },
    )
  elif args.command == 'process_batch':
    result = process_batch(
      args.csv_input,
      config = args.config,
      **{ key: value for key, value in vars(args).items() if '.' in key },
    )
