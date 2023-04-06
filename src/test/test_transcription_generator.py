import os

import pytest

from cli.pipeline_stages import transcription_generation as stage2


def extracted_audio_files():
    test_dir = "test/test_data/test_transcription_generator_data"
    audio_clip_paths = [path for path in os.listdir(test_dir) if path.endswith(".wav")]
    audio_and_transcript_paths = [(f"{test_dir}/{path}", f"{test_dir}/{path.replace('.wav', '.txt')}") for path in audio_clip_paths]
    return audio_and_transcript_paths


@pytest.mark.parametrize("extracted_audio_file, golden_transcript_file", extracted_audio_files())
def test_transcription_generation(extracted_audio_file, golden_transcript_file):
    transcript, label_probabilities, waveform_size = stage2.generate(extracted_audio_file, device="cpu")

    with open(golden_transcript_file) as f:
        golden_transcript = f.readline()

    split_transcript = transcript.split("|")
    split_golden_transcript = golden_transcript.split("|")
    total_number_of_words = len(golden_transcript)
    error_count = 0

    for (word, golden_word) in zip(split_transcript, split_golden_transcript):
        if word != golden_word:
            error_count += 1

    word_error_rate = error_count / total_number_of_words

    assert word_error_rate <= 0.1


def find_first_same_word(word, idx, transcript):
    for i in range(idx, len(transcript)):
        if transcript[i] == word:
            return i
    return None
