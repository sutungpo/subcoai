# -*- coding: utf-8 -*-

from src.subcoai.asr import load_model
from src.subcoai.asr import subs

device = "cuda"
audio_file = r"G:\Data\audio\023_0030_0205.mp3"
model_dir = "nemoasr"

m = load_model(model_dir, device=device, file_ext=".mp3")
result = m.transcribe(audio_file, chunk_size=20)
subs(result).save("output.srt")

""" batch_size = 2 # reduce if low on GPU mem
compute_type = "int8" # fp16 or fp32
# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("base", device,compute_type=compute_type)
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size, chunk_size=10)
subs(result).save("output_whisper.srt") """

""" 
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",    
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    input=r"G:\Data\audio\023_0030_0205.mp3",
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
 """
""" from reazonspeech.k2.asr import load_model, transcribe, audio_from_path

audio = audio_from_path(r"G:\Data\audio\023_0030_0205.mp3")
model = load_model()
ret = transcribe(model, audio)
print(ret.text) """
""" import torch
import nemo.collections.asr as nemo_asr
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt_ctc-0.6b-ja", map_location = torch.device('cuda'))
text = asr_model.transcribe([r"G:\Data\audio\023_0030_0245mono.mp3"])
print(text) """
