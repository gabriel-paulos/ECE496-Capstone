import os

import pytest

from cli.pipeline_stages import transcription_generation as stage2
from cli.pipeline_stages import timestamp_identification as stage3


def extracted_audio_files():
    test_dir = "test/test_data/test_timestamp_probability_score_data/"
    audio_clip_paths = [f"{test_dir}/{path}" for path in os.listdir(test_dir) if path.endswith(".wav")]
    return audio_clip_paths


@pytest.mark.parametrize("extracted_audio_file", extracted_audio_files())
def test_timestamp_probability_score(extracted_audio_file):

    transcript, label_probabilities, waveform_size = stage2.generate(extracted_audio_file, device="cpu")

    timestamps = stage3.identify(audio_file_path=extracted_audio_file, transcript=transcript,
                                 label_probabilities=label_probabilities, waveform_size=waveform_size,
                                 audio_offset=0, device="cpu")

    for timestamp in timestamps:
        assert timestamp.score >= 0.75
