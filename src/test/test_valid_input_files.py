import os

import pytest

from cli.pipeline_stages import audio_extraction as stage1


def extracted_acceptable_files():
    test_dir = "test/test_data/test_valid_input_files_data"
    paths = [f"{test_dir}/{path}" for path in os.listdir(test_dir) if path.endswith(".wav") or path.endswith(".mp4")]
    return paths


def extracted_unacceptable_files():
    test_dir = "test/test_data/test_valid_input_files_data"
    paths = [f"{test_dir}/{path}" for path in os.listdir(test_dir) if not path.endswith(".wav") and not path.endswith(".mp4")]
    return paths


@pytest.mark.parametrize("filename", extracted_acceptable_files())
def test_valid_input_files(filename):
    stage1.extract(filename, audio_chunk_size=10, no_split=False, output_filename=None)


@pytest.mark.parametrize("filename", extracted_unacceptable_files())
def test_invalid_input_files(filename):
    with pytest.raises(Exception) as e_info:
        stage1.extract(filename, audio_chunk_size=10, no_split=False, output_filename=None)
