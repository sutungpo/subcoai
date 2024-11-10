import os
import warnings
from abc import abstractmethod
from typing import List, Union, Optional, NamedTuple

# import ctranslate2
# import faster_whisper
import numpy as np
import torch
import torchaudio
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from .audio import SAMPLE_RATE, load_audio, log_mel_spectrogram
from .vad import load_vad_model, merge_chunks
from .types import TranscriptionResult, SingleSegment

import logging
logger = logging.getLogger(__name__)

""" 
class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    '''

    def generate_segment_batched(self, features: np.ndarray, tokenizer: faster_whisper.tokenizer.Tokenizer, options: faster_whisper.transcribe.TranscriptionOptions, encoder_output = None):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self.encode(features)

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )

        result = self.model.generate(
                encoder_output,
                [prompt] * batch_size,
                beam_size=options.beam_size,
                patience=options.patience,
                length_penalty=options.length_penalty,
                max_length=self.max_length,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
            )

        tokens_batch = [x.sequences_ids[0] for x in result]

        def decode_batch(tokens: List[List[int]]) -> str:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            # text_tokens = [token for token in tokens if token < self.eot]
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)

        return text

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = faster_whisper.transcribe.get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)
 """


class FasterPipeline(Pipeline):
    """
    abstract Pipeline wrapper for FasterAsrModel.
    """

    def __init__(self,
                 model,
                 vad,
                 vad_params: dict,
                 options: NamedTuple,
                 tokenizer=None,
                 device: Union[int, str, "torch.device"] = -1,
                 framework="pt",
                 language: Optional[str] = None,
                 suppress_numerals: bool = False,
                 **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(
            **kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def get_iterator(self, inputs, num_workers: int, batch_size: int,
                     preprocess_params, forward_params, postprocess_params):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 num_workers=num_workers,
                                                 batch_size=batch_size,
                                                 collate_fn=stack)
        model_iterator = PipelineIterator(dataloader,
                                          self.forward,
                                          forward_params,
                                          loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess,
                                          postprocess_params)
        return final_iterator

    def transcribe(self,
                   audio: Union[str, np.ndarray],
                   batch_size=None,
                   num_workers=0,
                   language=None,
                   task=None,
                   chunk_size=30,
                   skip_silence=True,
                   print_progress=False,
                   combined_progress=False) -> TranscriptionResult:
        audio = self.load_audio(audio)

        # audio is a pytoch tensor format
        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2]}

        vad_segments = self.vad_model({
            "waveform": audio.unsqueeze(0),
            "sample_rate": SAMPLE_RATE
        })
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            skip_silence=skip_silence,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        for idx, out in enumerate(
                self.__call__(data(audio, vad_segments),
                              batch_size=batch_size,
                              num_workers=num_workers)):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            text = out['text']
            # if batch_size in [0, 1, None]:
            #     text = text[0]
            segments.append({
                "text": text,
                "start": round(vad_segments[idx]['start'], 3),
                "end": round(vad_segments[idx]['end'], 3)
            })

        logger.info(segments)

        return {"segments": segments, "language": "ja"}

    @abstractmethod
    def load_audio(self, audio_path):
        raise NotImplementedError("load_audio not implemented")

class FasterFunPipeline(FasterPipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def preprocess(self, audio):
        return audio

    def _forward(self, model_inputs):
        outputs = self.model.inference(model_inputs['inputs'], **self.options)
        return {'text': outputs[0][0]['text']}

    def postprocess(self, model_outputs):
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        post_text = rich_transcription_postprocess(model_outputs["text"])
        return {'text': post_text}

    def load_audio(self, audio_path):
        from funasr.utils.load_utils import load_audio_text_image_video
        audio = load_audio_text_image_video(audio_path)
        return audio


class FasterK2Pipeline(FasterPipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """

    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def preprocess(self, audio):
        return audio

    def _forward(self, model_inputs):
        stream = self.model.create_stream()
        assert isinstance(model_inputs['inputs'], torch.Tensor)
        # k2speech model accept 1d tensor, not batch
        audio = model_inputs['inputs'].squeeze(0).numpy()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self.model.decode_stream(stream)
        return {'text': stream.result.text}

    def postprocess(self, model_outputs):
        return model_outputs

    def load_audio(self, audio_path):
        from reazonspeech.k2.asr import audio_from_path
        from reazonspeech.k2.asr.audio import audio_to_file, pad_audio, norm_audio
        audio = audio_from_path(audio_path)
        PAD_SECONDS = 0.9
        audio = pad_audio(norm_audio(audio), PAD_SECONDS)
        return torch.from_numpy(audio.waveform)


class FasterNemoPipeline(FasterPipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """

    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs
    def __init__(self,
                 model,
                 vad,
                 vad_params: dict,
                 options: NamedTuple,
                 tokenizer=None,
                 device: Union[int, str, "torch.device"] = -1,
                 framework="pt",
                 language: Optional[str] = None,
                 suppress_numerals: bool = False,
                 **kwargs):
        super().__init__(model, vad, vad_params, options, tokenizer, device,
                         framework, language, suppress_numerals, **kwargs)
        self.file_ext = kwargs.pop("file_ext", ".mp3")
        self.sample_rate = 16000

    def preprocess(self, audio):
        return audio

    def _forward(self, model_inputs, file_ext=None):
        audio = model_inputs['inputs']
        text = self.model.transcribe([audio])
        return {'text': text[0][0]}

    def postprocess(self, model_outputs):
        return model_outputs

    def load_audio(self, audio_path):
        import torchaudio.transforms as T
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        resample_transform = T.Resample(orig_freq=sample_rate,
                                        new_freq=SAMPLE_RATE)
        waveform = resample_transform(waveform)
        return waveform.squeeze(0)

    def get_iterator(self, inputs, num_workers: int, batch_size: int,
                     preprocess_params, forward_params, postprocess_params):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        # def stack(items):
        #     return {'inputs': torch.stack([x['inputs'] for x in items])}

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 num_workers=num_workers,
                                                 batch_size=batch_size)
        model_iterator = PipelineIterator(dataloader,
                                          self.forward,
                                          forward_params,
                                          loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess,
                                          postprocess_params)
        return final_iterator

    def transcribe(self,
                   audio: str,
                   batch_size=None,
                   num_workers=0,
                   language=None,
                   task=None,
                   chunk_size=30,
                   print_progress=False,
                   combined_progress=False) -> TranscriptionResult:
        audio_tensor = self.load_audio(audio)

        # audio is a pytoch tensor format
        def data(audio, segments):
            for idx, seg in enumerate(segments):
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                import subprocess
                import tempfile
                start_time = seg['start']
                end_time = seg['end']
                tmp_path =  tempfile.mkdtemp()
                out_audio = os.path.join(tmp_path, f"chunk_{idx}{self.file_ext}")
                command = [
                    "ffmpeg", "-i", f"{audio}", "-ss", f"{start_time:.2f}",
                    "-to", f"{end_time:.2f}", "-ar", f"{self.sample_rate}", "-ac", "1", f"{out_audio}"
                ]
                subprocess.run(command, check=True)

                yield {'inputs': out_audio}

        vad_segments = self.vad_model({
            "waveform": audio_tensor.unsqueeze(0),
            "sample_rate": SAMPLE_RATE
        })
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        for idx, out in enumerate(
                self.__call__(data(audio, vad_segments),
                              batch_size=batch_size,
                              num_workers=num_workers)):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            text = out['text']
            # if batch_size in [0, 1, None]:
            #     text = text[0]
            segments.append({
                "text": text,
                "start": round(vad_segments[idx]['start'], 3),
                "end": round(vad_segments[idx]['end'], 3)
            })

        return {"segments": segments, "language": self.preset_language}


class FasterM4tPipeline(FasterPipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """

    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs
    def __init__(self,
                 model,
                 vad,
                 vad_params: dict,
                 options: NamedTuple,
                 tokenizer=None,
                 device: Union[int, str, "torch.device"] = -1,
                 framework="pt",
                 language: Optional[str] = None,
                 suppress_numerals: bool = False,
                 **kwargs):
        super().__init__(model, vad, vad_params, options, tokenizer, device,
                         framework, language, suppress_numerals, **kwargs)
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")

    def preprocess(self, audio):
        audio = self.processor(audios=audio['inputs'], src_lang="jpn",return_tensors="pt")
        return {'inputs': audio}

    def _forward(self, model_inputs):
        tgt_lang = 'jpn'
        output_tokens = self.model.generate(**model_inputs['inputs'], tgt_lang=tgt_lang, generate_speech=False)
        translated_text_from_audio = self.processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
        return {'text': translated_text_from_audio}

    def postprocess(self, model_outputs):
        return model_outputs

    def load_audio(self, audio_path):
        audio, orig_freq = torchaudio.load(audio_path)
        audio = torchaudio.functional.resample(audio,
                                               orig_freq=orig_freq,
                                               new_freq=SAMPLE_RATE)
        return audio

def load_model(asr_arch,
               device,
               asr_options=None,
               language: Optional[str] = "auto",
               vad_model=None,
               vad_options=None,
               **kwargs):
    '''Load a Whisper model for inference.
    Args:
        asr_arch: str - The name of the Whisper model to load.
        device: str - The device to load the model on.
        compute_type: str - The compute type to use for the model.
        options: dict - A dictionary of options to use for the model.
        language: str - The language of the model. (use English for now)
        model: Optional[WhisperModel] - The WhisperModel instance to use.
        download_root: Optional[str] - The root directory to download the model to.
        threads: int - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    '''
    default_asr_options = {}
    match asr_arch:
        case "funasr":
            from funasr import AutoModel
            model, kwargs_m = AutoModel.build_model(
                model="iic/SenseVoiceSmall",
                trust_remote_code=True,
                remote_code="./model.py",
                device=device)
            model.eval()
            default_asr_options = {
                "language": language,
                "use_itn": False,
                "ban_emo_unk": False
            }
            default_asr_options.update(kwargs_m)
            inst_cl = FasterFunPipeline
        case "k2asr":
            from reazonspeech.k2.asr import load_model
            model = load_model(device=device)
            inst_cl = FasterK2Pipeline
        case "nemoasr":
            import nemo.collections.asr as nemo_asr
            model = nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt_ctc-0.6b-ja",
                map_location=torch.device(device))
            inst_cl = FasterNemoPipeline
        case "m4tasr":
            from transformers import SeamlessM4Tv2Model

            model = SeamlessM4Tv2Model.from_pretrained(
                "facebook/seamless-m4t-v2-large")
            model.to(device)
            inst_cl = FasterM4tPipeline
        case _:
            raise ValueError(f"Unsupported ASR model: {asr_arch}")

    if asr_options is not None:
        default_asr_options.update(asr_options)

    default_vad_options = {"vad_onset": 0.500, "vad_offset": 0.363}

    if vad_options is not None:
        default_vad_options.update(vad_options)

    if vad_model is not None:
        vad_model = vad_model
    else:
        vad_model = load_vad_model(torch.device(device),
                                   use_auth_token=None,
                                   **default_vad_options)

    return inst_cl(model=model,
                   vad=vad_model,
                   options=default_asr_options,
                   language=language,
                   vad_params=default_vad_options,
                   **kwargs)


def subs(result: TranscriptionResult, print_text=True):
    import pysubs2
    from pysubs2 import SSAFile, SSAEvent
    subs = SSAFile()
    for chunk in result['segments']:
        event = SSAEvent(start=pysubs2.make_time(s=chunk['start']),
                         end=pysubs2.make_time(s=chunk['end']))
        event.plaintext = chunk['text']
        subs.append(event)

    return subs
