import os
import random

import pytest

from cli.pipeline_stages import timestamp_identification as stage3


def extracted_audio_files():
    test_dir = "test/test_sample_data/test_transcript_lexeme_identification_data"
    audio_clip_paths = [path for path in os.listdir(test_dir) if path.endswith(".wav")]
    audio_and_transcript_paths = [(f"{test_dir}/{path}", f"{test_dir}/{path.replace('.wav', '.txt')}") for path in audio_clip_paths]
    return audio_and_transcript_paths


@pytest.mark.parametrize("extracted_audio_file, golden_transcript_file", extracted_audio_files())
def test_transcript_lexeme_identification(extracted_audio_file, golden_transcript_file):

    random.seed(10)

    with open(golden_transcript_file) as f:
        golden_transcript = f.readline()

    golden_path = stage3.identify(audio_file_path=extracted_audio_file, transcript=golden_transcript, audio_offset=0, device="cpu")

    sample_umm_lexemes = ["UM", "UMM", "UMMM", "UMMMMM", "UHM", "UHHHHM", "UHHM", "EUHM", "AM", "AMM", "AHM"]

    modified_transcript_words = golden_transcript.split("|")

    for i in range(len(modified_transcript_words)):
        if modified_transcript_words[i] in sample_umm_lexemes:
            modified_transcript_words[i] = random.choice(sample_umm_lexemes)

    modified_transcript = "|".join(modified_transcript_words)

    modified_path = stage3.identify(audio_file_path=extracted_audio_file, transcript=modified_transcript, audio_offset=0, device="cpu")

    assert len(golden_path) == len(modified_path)

    for i in range(len(golden_path)):
        assert modified_path[i].label in sample_umm_lexemes
        assert abs(golden_path[i].start - modified_path[i].start) <= 0.5
        assert abs(golden_path[i].end - modified_path[i].end) <= 0.5
