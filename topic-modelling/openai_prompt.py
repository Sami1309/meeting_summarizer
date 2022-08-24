import openai

openai.api_key = ''
prompt = '''

extract 2 decisions'''

response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=0.7,
            max_tokens=400,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
          )

print(response)