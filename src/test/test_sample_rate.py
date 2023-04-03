import os

import pytest

from cli.pipeline_stages import transcription_generation as stage2


def extracted_acceptable_audio_files():
    sample_dir = "test/test_sample_data/test_sample_rate_data"
    audio_clip_paths = [f"{sample_dir}/{path}" for path in os.listdir(sample_dir) if path.endswith("_G.wav")]
    return audio_clip_paths


def extracted_unacceptable_audio_files():
    sample_dir = "test/test_sample_data/test_sample_rate_data"
    audio_clip_paths = [f"{sample_dir}/{path}" for path in os.listdir(sample_dir) if path.endswith("_L.wav")]
    return audio_clip_paths


@pytest.mark.parametrize("extracted_audio_file", extracted_acceptable_audio_files())
def test_acceptable_sample_rate(extracted_audio_file):
    transcript, label_probabilities, waveform_size = stage2.generate(extracted_audio_file, device="cpu")



@pytest.mark.parametrize("extracted_audio_file", extracted_unacceptable_audio_files())
def test_unacceptable_sample_rate(extracted_audio_file):
    with pytest.raises(Exception) as e_info:
        transcript, label_probabilities, waveform_size = stage2.generate(extracted_audio_file, device="cpu")
