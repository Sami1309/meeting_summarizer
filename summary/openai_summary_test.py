import os
from click import prompt
import openai

dir = ""
openai.api_key = ""
files = ["AICoE_05122021.txt"]

chunk_len = 14000

for file in files:
    responses = []
    with open(dir+file, 'r') as f:
        prompt = f.read()
        prompt_len = len(prompt)
        chunk_num = prompt_len // chunk_len
        chunk_offset = prompt_len % chunk_len
        for i in range(chunk_num):
          response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt[i*chunk_len:(i+1)*chunk_len],
            temperature=0.7,
            max_tokens=400,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
          )
          responses.append(response)

        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt[:-chunk_offset],
            temperature=0.7,
            max_tokens=400,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
          )
        responses.append(response)

    with open(file[:-4]+"_summary.txt", 'w') as f:
      f.write(" ".join(response))