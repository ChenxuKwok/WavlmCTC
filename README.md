# WavlmCTC for Phoneme Recognition

This repository provides an implementation for phoneme recognition using the WavLM model. WavLM is available on Hugging Face, but it does not come with a pretrained tokenizer. As a result, you'll need to train a tokenizer manually. Additionally, due to some issues with loading the pretrained WavLM model directly from Hugging Face, we use [s3prl](https://github.com/s3prl/s3prl) as an alternative.

## Lexicon

The phoneme lexicon consists of the following:

* 39 CMU phonemes
* `<\blank>`: A special token used for CTC loss
* `<\unk>`: A special token for unknown phonemes

## Example Training Data

The training data should be in the following JSON format:

```json
{
    "id": "088_4075",
    "wav_file": "path/to/your.wav",
    "transcription": "W IH DH G R G R EY T K EH R"
}
```

* `"id"`: Unique identifier for the audio sample.
* `"wav_file"`: Path to the `.wav` file for the sample.
* `"transcription"`: Space-separated sequence of phonemes corresponding to the audio file.
