#!/usr/bin/env python

import click
from tqdm import tqdm

from pipeline_stages import audio_extraction as stage1
from pipeline_stages import transcription_generation as stage2
from pipeline_stages import timestamp_identification as stage3
from pipeline_stages import filler_word_removal as stage4

DEPTH = {
  "e": "EXTRACT_AUDIO",
  "g": "GENERATE_TIMESTAMSP",
  "i": "IDENTIFY_TIMESTAMPS",
  "f": "FULL",
}


@click.command()
@click.option("-b", "--bert", type=bool, show_default=True,
              default=False, help="Use BERT in timestamp identification")
@click.option("-c", "--chunk_size", type=int, show_default=True,
              default=20, help="Audio chunk size")
@click.option("-d", "--depth", show_default=True, default="f",
              type=click.Choice(DEPTH.keys()), help="Pipeline depth")
@click.option("-i", "--input", nargs=1, required=True,
              type=click.Path(exists=True), help="Input filename")
@click.option("-o", "--output", nargs=1,
              type=click.Path(exists=False), help="Output filename")
def main(bert, chunk_size, depth, input, output):

    input_filename = input
    output_filename = output
    audio_chunk_size = chunk_size
    run_pipeline_depth = DEPTH[depth]  # allowed values: EXTRACT_AUDIO, GENERATE_TRANSCRIPT, IDENTIFY_TIMESTAMPS, FULL
    use_bert = bert

    run_pipeline(input_filename, output_filename, audio_chunk_size=audio_chunk_size, run_pipeline_depth=run_pipeline_depth, use_bert=use_bert)


def run_pipeline(filename, output_filename=None, audio_chunk_size=20, run_pipeline_depth="FULL", use_bert=False):

    if run_pipeline_depth != "EXTRACT_AUDIO":
        offset_paths = stage1.extract(filename, audio_chunk_size=audio_chunk_size)
    else:
        offset_paths = stage1.extract(filename, no_split=True, output_filename=output_filename)
        print(f"Output audio file has been written to {offset_paths[0] if output_filename is None else output_filename}")
        return

    transcript_and_emissions = []
    for offset, path in tqdm(offset_paths, desc="Generating transcripts and extracting acoustic features"):
        transcript, emissions, waveform_size = stage2.generate(path)
        transcript_and_emissions.append((transcript, emissions, waveform_size, path, offset))

    with open("transcript.txt", "w") as f:
        for transcript, *_ in transcript_and_emissions:
            f.write(transcript)

    if run_pipeline_depth == "GENERATE_TRANSCRIPT":
        return

    timestamps = []

    for transcript, emissions, waveform_size, path, offset in tqdm(transcript_and_emissions, desc="Identifying timestamps of filler words"):
        words = stage3.identify(path, transcript, audio_offset=offset, label_probabilities=emissions, waveform_size=waveform_size, use_bert=use_bert)
        timestamps.extend(words)

    with open("timestamps.txt", "w") as f:
        for timestamp in timestamps:
            f.write(f"{timestamp.label} {timestamp.score} {timestamp.start} {timestamp.end}\n")

    if run_pipeline_depth == "IDENTIFY_TIMESTAMPS":
        return

    stage4.edit(filename, timestamps, output_filename=output_filename)


if __name__ == "__main__":
    main()
