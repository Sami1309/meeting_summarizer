from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from rouge import Rouge

rouge = Rouge()


tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")

model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-large")


prompt = '''   '''

tokens = tokenizer(prompt, truncation=True, padding="longest", return_tensors="pt")
summary = model.generate(**tokens)
pegasus_summary = tokenizer.decode(summary[0])
# print(f'ref summary : {ref_summary}\n')
print(f'pegasus summary : {pegasus_summary}\n')
# scores = rouge.get_scores(pegasus_summary, ref_summary)
# print(f'ROUGE : {scores}')



