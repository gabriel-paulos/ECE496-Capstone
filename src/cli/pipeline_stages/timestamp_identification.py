import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"

import re
from typing import Tuple, List

import numpy
import torch
import torchaudio
import transformers
import tensorflow as tf
from transformers import BertTokenizer, TFBertForMaskedLM
from torchaudio.pipelines import WAV2VEC2_ASR_LARGE_960H as WAV2VEC2
from torchaudio.functional import resample as resample

from .util.timestamp_identification_utils import Trellis, Segment, LabelPoint


def identify(audio_file_path,
             transcript,
             audio_offset,
             label_probabilities=None,
             waveform_size=None,
             default_model=WAV2VEC2.get_model(),
             labels=WAV2VEC2.get_labels(),
             model_sample_rate=WAV2VEC2.sample_rate,
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
             use_bert=False):

    if transcript is None or transcript == "":
        return []

    model = default_model.to(device)

    if label_probabilities is None or waveform_size is None:
        label_probabilities, waveform_size = get_label_probabilities(
                                                            device,
                                                            audio_file_path,
                                                            model,
                                                            model_sample_rate)

    processed_transcript = preprocess_transcript(transcript, labels)

    trellis = Trellis(processed_transcript, labels, label_probabilities)

    try:
        found_path = trellis.backtrack()
    except Exception:
        print(f"Alignment failure: {transcript}")
        return []

    segmented_path = merge_repeated_labels_in_path(found_path,
                                                   processed_transcript)
    boundary_segmented_path = merge_labels_by_boundary(segmented_path, "|")

    normalized_path = normalize_by_sampling_rate(boundary_segmented_path,
                                                 model_sample_rate,
                                                 waveform_size,
                                                 trellis.graph.size(0) - 1,
                                                 audio_offset)

    normalized_transcript = transcript.replace("|", " ")

    final_path = filter_filler_word(normalized_transcript, normalized_path, use_bert=use_bert)

    return final_path


def get_label_probabilities(device,
                            audio_file_path,
                            model,
                            model_sample_rate):

    with torch.inference_mode():
        waveform, audio_sample_rate = torchaudio.load(audio_file_path)
        if audio_sample_rate < 16000 or audio_sample_rate > 48000:
            raise ValueError(f"Sampling rate of {audio_sample_rate} is not within 16kHz to 48kHz allowed range!")
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
                               trellis_size,
                               audio_offset) -> List[Segment]:

    normalized_segmented_path = []

    normalization_constant = waveform_size / trellis_size

    for segment in segmented_path:
        start = int(segment.start * normalization_constant) / sampling_rate
        end = int(segment.end * normalization_constant) / sampling_rate
        normalized_segmented_path.append(Segment(
                                             segment.label,
                                             segment.score,
                                             start + audio_offset,
                                             end + audio_offset))

    return normalized_segmented_path


def get_topk_predictions(input, k, tokenizer, model):

    tokenized_inputs = tokenizer(input, return_tensors="tf")
    outputs = model(tokenized_inputs["input_ids"])

    top_k_indices = tf.math.top_k(outputs.logits, k).indices[0].numpy()
    decoded_output = tokenizer.batch_decode(top_k_indices)
    mask_token = tokenizer.encode(tokenizer.mask_token)[1:-1]
    mask_index = numpy.where(tokenized_inputs['input_ids'].numpy()[0] == mask_token)[0][0]

    decoded_output_words = decoded_output[mask_index]

    return decoded_output_words


def filter_filler_word(normalized_transcript, normalized_segmented_path,
                       analysis_tokenizer=BertTokenizer.from_pretrained("bert-large-cased"),
                       analysis_model=TFBertForMaskedLM.from_pretrained("bert-large-cased", return_dict=True),
                       use_bert=False):

    words_marked_for_deletion = []

    amm_match = re.compile("^AH*M+$")
    emm_match = re.compile("^EU*H*M*$")
    omm_match = re.compile("^OU*H*M*$")
    umm_match = re.compile("^UH*M*$")

    match_list = [amm_match, emm_match, omm_match, umm_match]

    for i, word in enumerate(normalized_segmented_path):

        if word.score < 0.75:
            continue

        if any(filter.match(word.label) for filter in match_list):

            if use_bert and any(filter.match(word.label) for filter in [amm_match, omm_match]):
                removed_predictions = get_topk_predictions(normalized_transcript.replace(word.label, "[MASK]"), 5, analysis_tokenizer, analysis_model)

                if word.label in removed_predictions:
                    continue

            if i != 0:
                word.start = normalized_segmented_path[i - 1].end

            if i != len(normalized_segmented_path) - 1:
                word.end = normalized_segmented_path[i + 1].start

            words_marked_for_deletion.append(word)

    return words_marked_for_deletion
