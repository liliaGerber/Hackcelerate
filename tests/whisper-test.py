
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",  # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    torch_dtype=torch.float16,
    device="cuda:0",  # or mps for Mac devices
    model_kwargs={"attn_implementation": "flash_attention_2"}
    if is_flash_attn_2_available()
    else {"attn_implementation": "sdpa"},
)

outputs = pipe(
    "audio.mp3",
    chunk_length_s=10,
    batch_size=48,
    return_timestamps=True,
)

print(outputs["text"])
"""
model_size = "large-v3-turbo"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="int8")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")


segments, info = model.transcribe("audio.mp3", beam_size=2)


# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

transcription = " ".join(segment.text.strip() for segment in segments)

print(transcription)
"""
