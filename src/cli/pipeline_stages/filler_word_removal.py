from datetime import timedelta

import moviepy.editor as mvpe


def edit(filename: str, timestamps, output_filename=None):

    edited_filename = output_filename if output_filename is not None else "FillerRemoved_" + filename

    if filename.endswith(".wav"):
        clip = mvpe.AudioFileClip(filename)
    elif filename.endswith(".mp4"):
        clip = mvpe.VideoFileClip(filename)

    for timestamp in timestamps:
        start = str(timedelta(seconds=timestamp.start))
        end = str(timedelta(seconds=timestamp.end))
        clip = clip.cutout(start, end)

    if filename.endswith(".wav"):
        clip.write_audiofile(filename=edited_filename)
    elif filename.endswith(".mp4"):
        clip.write_videofile(filename=edited_filename)
