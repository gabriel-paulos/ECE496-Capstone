import click
import moviepy.editor as mp

import os
import glob
from multiprocessing import get_context
from multiprocessing.pool import Pool

import transcription_generation.transcription_generation as stage1
import timestamp_identification.timestamp_identification as stage2

audio_filename = "ECE552-qa-03"


def task(i, path):
    transcript, emissions, waveform_size = stage1.generate(path)
    words = stage2.identify(path, transcript, label_probabilities=emissions, waveform_size=waveform_size)
    for word in words:
        print(word.label, word.score, word.start + i*20, word.end + i*20)


def main():

    clip = mp.VideoFileClip(audio_filename + ".mp4")

    for i in range(0, int(clip.duration), 20):
        subclip_end = min(i + 20, clip.duration)
        subclip = clip.subclip(i, subclip_end)
        subclip.audio.write_audiofile(f"sample_data/{audio_filename}_audio_part{i}.wav", verbose=False, logger=None)

    clip_paths = []

    for file in os.listdir("sample_data"):
        if file.startswith(f"{audio_filename}_audio_part") and file.endswith(".wav"):
            clip_paths.append(f"sample_data/{file}")

    for i, path in enumerate(clip_paths):
        task(i, path)

    #for f in glob.glob("sample_data/*.wav"):
    #    os.remove(f)


if __name__ == "__main__":
    main()
