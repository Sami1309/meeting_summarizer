import requests

subscription_key = ''


def get_token(subscription_key):
    fetch_token_url = 'https://eastus.api.cognitive.microsoft.com/sts/v1.0/issueToken'
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key
    }
    response = requests.post(fetch_token_url, headers=headers)
    access_token = str(response.text)
    print(access_token)

# get_token(subscription_key)

url = "https://eastus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-US"

headers = {
  'Content-type': 'audio/wav; codec=audio/pcm',
  'Ocp-Apim-Subscription-Key': '',
  'Connection': 'keep-alive',
  'Accept': 'application/json',
  'Expect' : '100-continue',
  'Authorization': ''
}

with open('xx.wav','rb') as payload:
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
