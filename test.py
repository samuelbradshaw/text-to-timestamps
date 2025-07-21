import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Internal imports
from text_to_timestamps.process import process_batch
from text_to_timestamps.benchmark import benchmark
import text_to_timestamps.utils


def main():
  text_to_timestamps.utils.write('Starting…\n\n')
  
  input_csv = 'sample/sample-input.tsv'
  config = 'sample/sample-config.json'
  process_batch(input_csv, config = config)


if __name__ == '__main__':
  main()
