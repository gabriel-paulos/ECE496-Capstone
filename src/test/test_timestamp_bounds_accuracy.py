import os

import pytest

from cli.pipeline_stages import transcription_generation as stage2
from cli.pipeline_stages import timestamp_identification as stage3


def extracted_audio_files():
    test_dir = "test/test_data/test_timestamp_bounds_accuracy_data"
    audio_clip_paths = [path for path in os.listdir(test_dir) if path.endswith(".wav")]
    audio_and_timestamp_paths = [(f"{test_dir}/{path}", f"{test_dir}/{path.replace('.wav', '.txt')}") for path in audio_clip_paths]
    return audio_and_timestamp_paths


@pytest.mark.parametrize("extracted_audio_file, golden_timestamps_file", extracted_audio_files())
def test_timestamp_bounds_accuracy(extracted_audio_file, golden_timestamps_file):

    transcript, label_probabilities, waveform_size = stage2.generate(extracted_audio_file, device="cpu")

    timestamps = stage3.identify(audio_file_path=extracted_audio_file, transcript=transcript,
                                 label_probabilities=label_probabilities, waveform_size=waveform_size,
                                 audio_offset=0, device="cpu")

    with open(golden_timestamps_file) as f:
        golden_timestamps_strs = f.readlines()

    golden_timestamps = []
    for timestamp_str in golden_timestamps_strs:
        timestamp_components = timestamp_str.split()
        timestamp = (float(timestamp_components[2]), float(timestamp_components[3]))
        golden_timestamps.append(timestamp)

    assert len(timestamps) == len(golden_timestamps)

    for timestamp, golden_timestamp in zip(timestamps, golden_timestamps):
        assert abs(timestamp.start - golden_timestamp[0]) <= 0.05
        assert abs(timestamp.end - golden_timestamp[1]) <= 0.05
