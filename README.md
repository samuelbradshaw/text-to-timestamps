# Text to Timestamps

**Text to Timestamps** (previously Text to VTT) is a Python package and command-line utility for aligning audio to a transcript. It can provide block, phrase, and/or word-level timestamps for an existing transcript or a generated transcript.

This tool provides a single interface for interacting with a wide variety of related open-source machine learning libraries. Tasks can be processed individually, or in bulk with CSV input. Timestamps can be output in CSV, JSON, or WebVTT formats.

Example output files can be found in the [sample output](https://samuelbradshaw.github.io/text-to-timestamps/sample/output) folder in this repository. There is also a [Timestamps Preview](https://samuelbradshaw.github.io/text-to-timestamps/sample/preview.html) page for visualizing and validating the accuracy of the output.


### Documentation:

- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install via PyPI](#install-via-pypi)
    - [For local development](#for-local-development)
- [Usage](#usage)
    - [Command line](#command-line)
    - [Python package](#python-package)
- [Configuration](#configuration)
    - [Configuration options](#configuration-options)


## <a name="installation"></a>Installation

### <a name="prerequisites"></a>Prerequisites

Text to Timestamps was tested on a Mac with Apple silicon, but it should work on other platforms as well.

- Python 3.10, 3.11, or 3.12
- Homebrew package manager (or an equivalent, if not on macOS)

### <a name="install-via-pypi"></a>Install via PyPI

This is the simplest way to install Text to Timestamps:

```bash
brew install ffmpeg sound-touch rust
pip install text-to-timestamps
```

### <a name="for-local-development"></a>For local development

You can also install an editable version for local development.
```bash
brew install ffmpeg sound-touch rust
git clone https://github.com/samuelbradshaw/text-to-timestamps text-to-timestamps
cd text-to-timestamps
python -m venv .venv
. .venv/bin/activate
pip install --editable .
```

You can run `test.py` as a script like this:
```bash
. .venv/bin/activate
python test.py
```


## <a name="usage"></a>Usage

### <a name="command-line"></a>Command line

Two functions are available via command line: `process` and `process_batch`.

---
**`process`**

**--job_id** – Required. A number or string that represents the processing task. This will be used in the output filename. If you are processing several audio files, each job ID should be unique to avoid naming conflicts. Examples: "1-corinthians-13", "dQw4w9WgXcQ", "My Cool Song", "25"

**--lang** – Required. BCP 47 language tag for the content being processed. Examples: "en", "fr", "ceb"

**--audio** – Required. URL or file path to the audio file to be processed. If a URL is provided, the video will be downloaded. Several file formats are supported, including WAV, MP3, etc. Video MP4 files are also supported (but not YouTube URLs).

**--transcript** – Optional. File path or string containing an existing transcript of the audio in plain text format. If this is not provided, Text to Transcript will attempt to generate a transcript. Paragraphs should be separated with two line breaks. For convenience, the escape sequence "\n" will be converted to a line break on import.

**--config** – Optional. File path or JSON string with configuration options. If providing a JSON string, it must be surrounded by single quotes. Any single quotes or new lines in the JSON string will need to be escaped. Config options can also be passed in as separate arguments. Any config options that aren't specified will fall back to [default config](https://github.com/samuelbradshaw/text-to-timestamps/blob/main/src/text_to_timestamps/default_config.json) values. See "Configuration options" below for a description of each option. Example: [sample-config.json](https://github.com/samuelbradshaw/text-to-timestamps/blob/main/sample/sample-config.json)

Examples:
```bash
% text_to_timestamps process \
  --job_id '1-corinthians-13' \
  --lang 'en' \
  --audio 'https://media2.ldscdn.org/assets/scriptures/the-new-testament/2015-11-1460-1-corinthians-13-male-voice-64k-eng.mp3' \
  --transcript '/Users/sbradshaw/projects/1-corinthians-13.txt' \
  --config '/Users/sbradshaw/projects/config.json'
```
```bash
% text_to_timestamps process \
  --job_id 'one-small-step' \
  --lang 'en' \
  --audio 'https://www.nasa.gov/wp-content/uploads/2015/01/590331main_ringtone_smallStep.mp3' \
  --general.workfiles_directory '/Users/sbradshaw/projects' \
  --general.output_directory '/Users/sbradshaw/projects' \
  --general.voice_isolation_method 'noisereduce' \
  --transcribe.method 'pywhispercpp'
```

---
**`process_batch`**

**--input_csv** – Required. File path to a CSV (or TSV) input file, where each row is a processing task. Example: [sample-input.tsv](https://github.com/samuelbradshaw/text-to-timestamps/blob/main/sample/sample-input.tsv)

**--config** – Optional. Same as above.

Examples:
```bash
% text_to_timestamps process_batch \
  --input_csv '/Users/sbradshaw/projects/input.tsv'
```
```bash
% text_to_timestamps process_batch \
  --input_csv '/Users/sbradshaw/projects/input.tsv' \
  --config '/Users/sbradshaw/projects/config.json'
```

### <a name="python-package"></a>Python package

The following functions are available: `process()`, `process_batch()`, `prepare()`, `transcribe()`, `get_words()`, `sort_words()`, `output_to_file()`.

---
**`process(job_id, lang, audio, transcript = None, config = text_to_timestamps.utils.default_config, **config_args)`**

Processes a given audio file from start to finish, including voice detection, transcription, and/or alignment (as specified in the config), and outputs a CSV, JSON, or WebVTT file with timestamp data. Returns timestamp data.

**job_id** – Required. A number or string that represents the processing task. This will be used in the output filename. If you are processing several audio files, each job ID should be unique to avoid naming conflicts. Examples: "1-corinthians-13", "dQw4w9WgXcQ", "My Cool Song", "25"

**lang** – Required. BCP 47 language tag for the content being processed. Examples: "en", "fr", "ceb"

**audio** – Required. URL or file path to the audio file to be processed. If a URL is provided, the video will be downloaded. Several file formats are supported, including WAV, MP3, etc. Video MP4 files are also supported (but not YouTube URLs).

**transcript** – Optional. File path or string containing an existing transcript of the audio in plain text format. If this is not provided, Text to Transcript will attempt to generate a transcript. Paragraphs should be separated with two line breaks. For convenience, the escape sequence "\n" will be converted to a line break on import.

**config** – Optional. File path, JSON string, or Python dict with configuration options. Any config options that aren't specified will fall back to [default config](https://github.com/samuelbradshaw/text-to-timestamps/blob/main/src/text_to_timestamps/default_config.json) values. See "Configuration options" below for a description of each option. Example: [sample-config.json](https://github.com/samuelbradshaw/text-to-timestamps/blob/main/sample/sample-config.json)

Examples:
```python
from text_to_timestamps.process import process

# Process with default configuration and no transcript
audio_url = 'https://www.nasa.gov/wp-content/uploads/2015/01/590331main_ringtone_smallStep.mp3'
process('one-small-step-1', 'en', audio_url)
```
```python
from text_to_timestamps.process import process

# Process with custom configuration path and transcript path
audio_url = 'https://www.nasa.gov/wp-content/uploads/2015/01/590331main_ringtone_smallStep.mp3'
transcript_path = '/Users/sbradshaw/projects/one-small-step.txt'
config_path = '/Users/sbradshaw/projects/config.json'
process('one-small-step-2', 'en', audio_url, transcript = transcript_path, config = config_path)
```
```python
from text_to_timestamps.process import process

# Process with custom configuration dict and transcript text
audio_url = 'https://www.nasa.gov/wp-content/uploads/2015/01/590331main_ringtone_smallStep.mp3'
transcript = 'OK, I’m going to step off the LM now.\nThat’s one small step for a man, one giant leap for mankind.'
config = {
  'general': {
    'workfiles_directory': '/Users/sbradshaw/projects',
    'output_directory': '/Users/sbradshaw/projects',
    'voice_isolation_method': 'noisereduce',
  },
}
process('one-small-step-3', 'en', audio_url, transcript = transcript, config = config)
```

---
**`process_batch(input_csv, config = None, **config_args)`**

Processes a batch of audio files from start to finish, including voice detection, transcription, and/or alignment (as specified in the config), using a CSV input file. Outputs CSV, JSON, or WebVTT files with timestamp data. Returns timestamp data for all jobs.

**input_csv** – Required. File path to a CSV (or TSV) input file, where each row is a processing task. Example: [sample-input.tsv](https://github.com/samuelbradshaw/text-to-timestamps/blob/main/sample/sample-input.tsv)

**config** – Optional. Same as above.

Examples:
```python
from text_to_timestamps.process import process_batch

# Process batch with default configuration (or configuation specified in the CSV)
input_csv_path = '/Users/sbradshaw/projects/input.csv'
process_batch(input_csv_path)
```
```python
from text_to_timestamps.process import process_batch

# Process batch with specified config
input_csv_path = '/Users/sbradshaw/projects/input.csv'
config = {
  'general': {
    'workfiles_directory': '/Users/sbradshaw/projects',
    'output_directory': '/Users/sbradshaw/projects',
    'voice_isolation_method': 'noisereduce',
  },
}
process(input_csv_path, config = config)
```

---
**`prepare(job_id, audio, workfiles_directory, voice_isolation_method = None, force = False)`**

Downloads or copies the audio to the workfiles directory, converts it to WAV, and applies voice isolation if needed. Returns the path of the prepared audio file.

**job_id** – Required. Same as above.

**audio** – Required. Same as above.

**workfiles_directory** – Required. Path to a folder where Text to Timestamps can copy or download audio files for processing.

**voice_isolation_method** – Optional. Method for voice isolation or stem separation. Default: None.

**force** – Optional. Whether previously-processed files in the workfiles directory with the same filename should be overwritten (True) or skipped (False). When set to True, media will be downloaded or copied into the workfiles directory each time the script runs. Default: False.

---
**`transcribe(method, audio_path, lang = 'en', get_words_if_available = False, model_size = 'small', model_name = None, batch_size = 16)`**

Transcribes a given audio file (speech to text), using the specified transcription method. Returns a dict with the transcribed text, timestamped words (if applicable), and full transcription data (varies depending on the transcription method).

**method** – Required. Method for transcription.

**audio_path** – Required. File path to the audio file.

**lang** – Optional. BCP 47 language tag for the content being processed. If not specified, "en" (English) will be used.

**get_words_if_available** – Optional. Whether the transcription method should provide word-level timestamps, if the transcription method supports it. If word-level timestamps aren't available, they can be fetched later by passing the transcript into `get_words()`. Default: False.

**model_size** – Optional. The size of the machine learning model. Smaller models are generally less accurate, but they take less disk space and processing power. Default: "small".

**model_name** – Optional. The name of the machine learning model to use, if known. Different methods use different model names. If not provided, `model_size` will be used to choose the best model. Default: None.

**batch_size** – Optional. Batch size, or how many processes should be run at once. Larger batch sizes are usually faster, but they demand more resources. Not all methods support batching, so this will be ignored if not applicable. Default: 16.

---
**`get_words(method, audio_path, transcript, lang = 'en', model_size = 'small', model_name = None, batch_size = 16)`**

Breaks up a transcript into individual timestamped words, using the specified alignment method. Returns a list of timestamped words.

**method** – Required. Method for alignment.

**audio_path** – Required. File path to the audio file.

**transcript** – Required. Transcribed text content, from an existing source or generated using `transcribe()`.

**lang** – Optional. BCP 47 language tag for the content being processed. If not specified, "en" (English) will be used.

**model_size** – Optional. Same as above.

**model_name** – Optional. Same as above.

**batch_size** – Optional. Same as above.

---
**`sort_words(unsorted_words, transcript, output_config = {})`**

Adds metadata to the timestamped words and sorts them by block and phrase. Returns a dict with words, words by phrase, and words by block.

**unsorted_words** – Required. A list of timestamped words from `transcribe()` or `get_words()`. Each word should be a Python dictionary with keys `text`, `start`, and `end`.

**transcript** – Required. Transcribed text content in its original form.

**output_config** – Optional. Output configuration options. Any options not specified will fall back to defaults.

---
**`output_to_file(job_id, sorted_words, output_directory, output_config = {}, additional_metadata = {}, force = False)`**

Writes the timestamp data to a CSV, JSON, or WebVTT file.

**job_id** – Required. Same as above.

**sorted_words** – Required. Output from `sort_words()`.

**output_directory** – Required. Path to a folder where output files will be written.

**output_config** – Optional. Same as above.

**additional_metadata** – Optional. Additional metadata to include at the top of the output file (for WebVTT and JSON only).

**force** – Optional. Whether previously-generated files in the output directory with the same filename should be overwritten (True) or skipped (False). Default: True.

---
Combined example:
```python
from text_to_timestamps.prepare import prepare
from text_to_timestamps.transcribe import transcribe
from text_to_timestamps.align import get_words, sort_words
from text_to_timestamps.output import output_to_file

job_id = 'one-small-step'
lang = 'en'
output_directory = '/Users/sbradshaw/projects'
output_config = {
  'format': ['json'],
  'granularity': ['phrase', 'word'],
  'break_phrases_at: ['sentence_punctuation'],
}

# Prepare audio file
workfiles_directory = '/Users/sbradshaw/projects'
audio_url = 'https://www.nasa.gov/wp-content/uploads/2015/01/590331main_ringtone_smallStep.mp3'
audio_path = prepare(job_id, audio_url, workfiles_directory, voice_isolation_method = 'noisereduce')

# Transcribe
transcription_result = transcribe('whisper-jax', audio_path, lang = lang, get_words_if_available = False, model_size = 'small')

# Align
transcript = transcription_result['text']
words = get_words('forcealign-align', audio_path, transcript, lang = lang, model_size = 'small')
sorted_words = sort_words(words, transcript, output_config = output_config)

# Output
additional_metadata = { 'id': job_id, }
output_to_file(job_id, sorted_words, output_directory, output_config = output_config, additional_metadata = additional_metadata)
```


## <a name="configuration"></a>Configuration

When using the high-level `process()` and `process_batch()` functions, configuration options can be passed in as a JSON file path, a JSON string, or (Python only) a Python dict. Any options that aren't set will fall back to [default config](https://github.com/samuelbradshaw/text-to-timestamps/blob/main/src/text_to_timestamps/default_config.json) values.

`process_batch()` also accepts configuration options as columns in the CSV input (for example, adding a column `general.voice_isolation_method` allows you to specify a different voice isolation setting for each row. Options in the CSV take priority over JSON configuration. See [sample-input.tsv](https://github.com/samuelbradshaw/text-to-timestamps/blob/main/sample/sample-input.tsv) for an example.

The lower-level functions `prepare()`, `transcribe()`, `get_words()`, `sort_words()`, and `output_to_file()` are configured by passing in arguments directly, rather than relying on global configuration options.


### <a name="configuration-options"></a>Configuration options

- **general.workfiles_directory** – Path to a folder where Text to Timestamps can copy or download audio files for processing. Default: "text_to_timestamps_workfiles".

- **general.output_directory** – Path to a folder where output files will be written. Default: "text_to_timestamps_output".

- **general.overwrite_previous_workfiles** – Whether previously-processed files in the workfiles directory with the same filename should be overwritten (True) or skipped (False). When set to True, media will be downloaded or copied into the workfiles directory each time the script runs. Default: False.

- **general.overwrite_previous_output** – Whether previously-generated files in the output directory with the same filename should be overwritten (True) or skipped (False). Default: True.

- **general.voice_isolation_method** – Method for voice isolation or stem separation. Supported voice isolation methods can be found in [utils.py](https://github.com/samuelbradshaw/text-to-timestamps/blob/main/src/text_to_timestamps/utils.py). Default: None.

- **transcribe.method** – Method for transcription. Supported transcribe methods can be found in [utils.py](https://github.com/samuelbradshaw/text-to-timestamps/blob/main/src/text_to_timestamps/utils.py). Default: "stable-ts-faster-whisper".

- **transcribe.model_size** – The size of the machine learning model for transcription. Smaller models are generally less accurate, but they take less disk space and processing power. Default: "small".

- **transcribe.model_name** – The name of the machine learning model to use for transcription, if known. Different methods use different model names. If not provided, `transcribe.model_size` will be used to choose the best model. Default: None.

- **transcribe.batch_size** - Batch size, or how many processes should be run at once. Larger batch sizes are usually faster, but they demand more resources. Not all methods support batching, so this will be ignored if not applicable. Default: 16.

- **align.method** – Method for alignment. Supported align methods can be found in [utils.py](https://github.com/samuelbradshaw/text-to-timestamps/blob/main/src/text_to_timestamps/utils.py). Default: "stable-ts-faster-whisper-align".

- **align.model_size** – The size of the machine learning model for alignment. Smaller models are generally less accurate, but they take less disk space and processing power. Default: "small".

- **align.model_name** – The name of the machine learning model to use for alignment, if known. Different methods use different model names. If not provided, `align.model_size` will be used to choose the best model. Default: None.

- **align.batch_size** - Batch size, or how many processes should be run at once. Larger batch sizes are usually faster, but they demand more resources. Not all methods support batching, so this will be ignored if not applicable. Default: 16.

- **output.format** – List of output formats. Supported values: "csv", "json", "webvtt". Default: ["csv"].

- **output.granularity** – List specifying the granularity of timestamps in output files. A separate file will be output for each granularity. Supported values: "block", "phrase", "word". Default: ["word"].

- **output.break_phrases_at** – List specifying how the transcript should be broken up into phrases. Supported values: "single_line_breaks", "character_count", "sentence_punctuation". Default: ["single_line_breaks", "character_count"].

- **output.max_phrase_character_count** – Maximum number of characters in a phrase. Only applicable when `output.break_phrases_at` includes "character_count". Default: 40.

- **output.time_offset** – Number of seconds that should be added or removed from each calculated timestamp. Default: 0.

- **output.csv_delimiter** – Field delimiter that should be used in CSV output, such as "," or "\t". Default: ",".

- **output.json_indent** – Indentation in JSON output. If set to 0 or None, the JSON will be minified. Default: 2.

- **output.json_as_js** – Whether to output the JSON in a JavaScript file so it can be imported into an HTML document using the `<script>` tag. Default: False.

- **output.webvtt_timestamp_tags** – Whether [timestamp tags](https://developer.mozilla.org/en-US/docs/Web/API/WebVTT_API/Web_Video_Text_Tracks_Format#cue_payload_text_tags) should be added before each word in WebVTT output. This is useful for karaoke and other follow-along applications, but may not be supported in all contexts that use WebVTT. Default: False.

- **output.include_text** – Whether the text from the transcript should be included in output files. Default: True.

- **output.include_prefixes** – Whether unspoken content before each segment (such as whitespace, verse numbers, etc.) should be included in output files. Default: True.

- **output.include_suffixes** – Whether unspoken content after each segment (such as whitespace) should be included in output files. Default: True.

- **output.include_start_times** – Whether start times for each segment should be included in output files. Default: True.

- **output.include_end_times** – Whether end times for each segment should be included in output files. Default: True.

- **output.include_start_offsets** – Whether character start offsets for each segment should be included in output files. Default: False.

- **output.include_end_offsets** – Whether character end offsets for each segment should be included in output files. Default: False.

- **output.include_block_numbers** – Whether block numbers should be included in output files. Default: False.

- **output.include_phrase_numbers** – Whether phrase numbers should be included in output files. Default: False.

- **output.include_word_numbers** – Whether word numbers should be included in output files. Default: False.
