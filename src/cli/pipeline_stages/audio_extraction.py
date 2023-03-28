import os
import tempfile

import moviepy.editor as mvpe
import proglog
from tqdm import tqdm


def extract(filename: str, audio_chunk_size=20, no_split=False, output_filename=None):

    acceptable_video_file_formats = [".mp4"]

    if not any(filename.endswith(file_extension) for file_extension in acceptable_video_file_formats):
        raise ValueError(f"Filename not in list of acceptable formats! Acceptable file formats: {acceptable_video_file_formats}")

    clip = mvpe.VideoFileClip(filename)

    subclip_paths = []

    if no_split:
        output_filename = f"{os.path.splitext(filename)[0]}.wav" if output_filename is None else output_filename
        clip.audio.write_audiofile(output_filename, logger=ExtractAudioLogger(leave_bars=True, print_messages=False))
        return [output_filename]

    for i in tqdm(range(0, int(clip.duration), audio_chunk_size), desc="Extracting audio"):
        subclip_end = min(i + audio_chunk_size, int(clip.duration))
        subclip = clip.subclip(i, subclip_end)
        audio_file = tempfile.NamedTemporaryFile(prefix=f"{i}_", suffix=".wav")
        subclip.audio.write_audiofile(audio_file.name, logger=None)
        subclip_paths.append((i, audio_file))

    return subclip_paths


class ExtractAudioLogger(proglog.TqdmProgressBarLogger):
    def bars_callback(self, bar, attr, value, old_value=None):
        self.bars[bar]["title"] = "Extracting audio"
        super().bars_callback(bar, attr, value, old_value)
