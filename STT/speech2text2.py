import torch 
from transformers import Speech2Text2Processor, Wav2Vec2Processor, SpeechEncoderDecoderModel, Wav2Vec2Tokenizer, Speech2Text2Tokenizer
# from datasets import load_dataset 
import soundfile as sf 
from scipy.io import wavfile

# import librosa

model = SpeechEncoderDecoderModel.from_pretrained("./s2t-wav2vec2-large-en-de")

tokenizer = Wav2Vec2Processor.from_pretrained("./s2t-wav2vec2-large-en-de")
# processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# ds = ds.map(map_to_array)

# inputs = processor(ds["speech"][0], sampling_rate=16_000, return_tensors="pt")
# speech, rate = librosa.load("/Users/keltonzhang/Desktop/Dell_Proj/amicorpus/ES2008a/audio/ES2008a.Mix-Headset16khz.wav")

# from pydub import AudioSegment
# speech = AudioSegment.from_wav("/Users/keltonzhang/Desktop/Dell_Proj/amicorpus/ES2008a/audio/ES2008a.Mix-Headset16khz.wav")

# inputs = processor(speech, sampling_rate=16_000, return_tensors="pt")
# generated_ids = model.generate(inputs=inputs["input_values"], attention_mask=inputs["attention_mask"])

# transcription = processor.batch_decode(generated_ids)
rate, speech = wavfile.read("/Users/keltonzhang/Desktop/Dell_Proj/amicorpus/ES2008a/audio/ES2008a.Mix-Headset16khz.wav")
input_values = tokenizer(speech, return_tensors = 'pt', sampling_rate = rate).input_values
print(type(input_values))
logits = model(input_values.float()).logits
# model.train(per_device_train_batch_size=2)
# predicted_ids = torch.argmax(logits, dim =-1)
# transcriptions = tokenizer.decode(predicted_ids[0])
# print(transcriptions)