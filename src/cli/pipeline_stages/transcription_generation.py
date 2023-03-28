import torch
import torchaudio

from torchaudio.pipelines import WAV2VEC2_ASR_LARGE_960H as WAV2VEC2
from torchaudio.functional import resample as resample

from .util.transcription_generation_utils import GreedyCTCDecoder


def generate(audio_file_path,
             default_model=WAV2VEC2.get_model(),
             labels=WAV2VEC2.get_labels(),
             model_sample_rate=WAV2VEC2.sample_rate):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = default_model.to(device)

    label_probabilities, waveform_size = get_label_probabilities(
                                                device,
                                                audio_file_path,
                                                model,
                                                model_sample_rate)

    decoder = GreedyCTCDecoder(labels)
    transcript = decoder(label_probabilities)

    return transcript, label_probabilities, waveform_size


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
