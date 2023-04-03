import os

import pytest

from cli.pipeline_stages import transcription_generation as stage2


def extracted_audio_files():
    sample_dir = "test/test_sample_data/test_transcription_generator_data"
    audio_clip_paths = [path for path in os.listdir(sample_dir) if path.endswith(".wav")]
    audio_and_transcript_paths = [(f"{sample_dir}/{path}", f"{sample_dir}/{path.replace('.txt', '.wav')}") for path in audio_clip_paths]
    return audio_and_transcript_paths


@pytest.mark.parametrize("extracted_audio_file, golden_transcript", extracted_audio_files())
def test_transcription_generation(extracted_audio_file, golden_transcript):
    transcript, label_probabilities, waveform_size = stage2.generate(extracted_audio_file, device="cpu")

    total_number_of_words = len(golden_transcript.split("|"))
    error_count = 0
    for (word, golden_word) in zip(transcript.split("|"), golden_transcript.split("|")):
        if word != golden_word:
            error_count += 1

    word_error_rate = error_count / total_number_of_words

    assert word_error_rate <= 0.05
