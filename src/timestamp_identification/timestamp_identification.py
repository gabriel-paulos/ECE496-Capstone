import torch
import numpy
import torchaudio

from typing import Tuple, List

from .util.LabelPoint import LabelPoint, Segment
from .util.Trellis import Trellis

from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H as WAV2VEC2
from torchaudio.functional import resample as resample


def identify(audio_file_path,
             transcript,
             default_model=WAV2VEC2.get_model(),
             labels=WAV2VEC2.get_labels(),
             model_sample_rate=WAV2VEC2.sample_rate):

    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = default_model.to(device)

    label_probabilities, waveform_size = get_label_probabilities(
                                                            device,
                                                            audio_file_path,
                                                            model,
                                                            model_sample_rate)

    processed_transcript = preprocess_transcript(transcript, labels)

    trellis = Trellis(processed_transcript, labels, label_probabilities)

    found_path = trellis.backtrack()

    segmented_path = merge_repeated_labels_in_path(found_path,
                                                   processed_transcript)

    boundary_segmented_path = merge_labels_by_boundary(segmented_path, "|")

    final_path = normalize_by_sampling_rate(boundary_segmented_path,
                                            model_sample_rate,
                                            waveform_size,
                                            trellis.graph.size(0) - 1)

    return final_path


def get_label_probabilities(device,
                            audio_file_path,
                            model,
                            model_sample_rate):

    with torch.inference_mode():
        waveform, audio_sample_rate = torchaudio.load(audio_file_path)
        waveform = resample(waveform, audio_sample_rate, model_sample_rate)
        label_logits, _ = model(waveform.to(device))
        label_probabilities = torch.log_softmax(label_logits, dim=-1)

    return label_probabilities[0].cpu().detach(), waveform.size(1)


def preprocess_transcript(transcript: str, labels: Tuple[str]) -> str:

    bar_transcript = transcript.replace(" ", "|").replace("||", "|")
    token_only_transcript = "".join([c for c in bar_transcript if c in labels])
    final_transcript = token_only_transcript.upper()

    return final_transcript


def merge_repeated_labels_in_path(path: List[LabelPoint],
                                  transcript: str) -> List[Segment]:

    segmented_path = []

    i, j = 0, 0

    while i < len(path):
        while j < len(path) and path[i].token_index == path[j].token_index:
            j += 1

        averaged_score = sum(path[k].score for k in range(i, j)) / (j - i)
        segmented_path.append(
                             Segment(
                                    transcript[path[i].token_index],
                                    averaged_score,
                                    path[i].time_index,
                                    path[j - 1].time_index + 1))
        i = j

    return segmented_path


def merge_labels_by_boundary(segmented_path: List[Segment],
                             boundary: str) -> List[Segment]:

    boundary_segmented_path = []

    i, j = 0, 0

    while i < len(segmented_path):
        if j >= len(segmented_path) or segmented_path[j].label == boundary:
            if i != j:
                segments = segmented_path[i:j]
                word_label = "".join([segment.label for segment in segments])
                weighted_word_score = sum(
                    segment.score * len(segment) for segment in segments) \
                    / sum(len(segment) for segment in segments)
                boundary_segmented_path.append(
                                            Segment(word_label,
                                                    weighted_word_score,
                                                    segmented_path[i].start,
                                                    segmented_path[j - 1].end))
            i = j + 1
            j = i
        else:
            j += 1

    return boundary_segmented_path


def normalize_by_sampling_rate(segmented_path: List[Segment],
                               sampling_rate,
                               waveform_size,
                               trellis_size) -> List[Segment]:

    normalized_segmented_path = []

    normalization_constant = waveform_size / trellis_size

    for segment in segmented_path:
        start = int(segment.start * normalization_constant) / sampling_rate
        end = int(segment.end * normalization_constant) / sampling_rate
        normalized_segmented_path.append(Segment(
                                             segment.label,
                                             segment.score,
                                             start,
                                             end))

    return normalized_segmented_path
