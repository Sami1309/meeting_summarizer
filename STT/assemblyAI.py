api_key = ""

import requests
import timeit

start = timeit.default_timer()
filename = "xx.mp3"
def read_file(filename, chunk_size=5242880):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data

headers = {'authorization': api_key}
response = requests.post('https://api.assemblyai.com/v2/upload',
                        headers=headers,
                        data=read_file(filename))

json = response.json()
audio_url = json["upload_url"]
endpoint = "https://api.assemblyai.com/v2/transcript"
json = {
    "audio_url": audio_url,
    "speaker_labels": True,
    "auto_highlights": True
}
headers = {
    "authorization": api_key,
    "content-type": "application/json"
}
response = requests.post(endpoint, json=json, headers=headers)
print(response.json())

end = timeit.default_timer()
diff = end - start
print(diff)