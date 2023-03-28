import click
from tqdm import tqdm

from pipeline_stages import audio_extraction as stage1
from pipeline_stages import transcription_generation as stage2
from pipeline_stages import timestamp_identification as stage3


def main():

    # put cli code here, some relevant options to capture below

    video_filename = "../test_file.mp4"
    # output_filename = "../test_file_edited.mp4"
    audio_chunk_size = 20
    run_pipeline_depth = "FULL"  # allowed values: EXTRACT_AUDIO, GENERATE_TRANSCRIPT, IDENTIFY_TIMESTAMPS, FULL

    run_pipeline(video_filename, audio_chunk_size=audio_chunk_size, run_pipeline_depth=run_pipeline_depth)


def run_pipeline(video_filename, output_filename=None, audio_chunk_size=20, run_pipeline_depth="FULL"):

    if run_pipeline_depth != "EXTRACT_AUDIO":
        offset_paths = stage1.extract(video_filename, audio_chunk_size=audio_chunk_size)
    else:
        offset_paths = stage1.extract(video_filename, no_split=True, output_filename=output_filename)
        print(f"Output audio file has been written to {offset_paths[0] if output_filename is None else output_filename}")
        return

    transcript_and_emissions = []
    for offset, path in tqdm(offset_paths, desc="Generating transcripts and extracting acoustic features"):
        transcript, emissions, waveform_size = stage2.generate(path)
        transcript_and_emissions.append((transcript, emissions, waveform_size, path, offset))

    if run_pipeline_depth == "GENERATE_TRANSCRIPT":
        return

    for transcript, emissions, waveform_size, path, offset in tqdm(transcript_and_emissions, desc="Identifying timestamps of filler words"):
        words = stage3.identify(path, transcript, audio_offset=offset, label_probabilities=emissions, waveform_size=waveform_size)
        for word in words:
            print(word.label, word.score, word.start, word.end)

    if run_pipeline_depth == "IDENTIFY_TIMESTAMPS":
        return


if __name__ == "__main__":
    main()
