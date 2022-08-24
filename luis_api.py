########### Python 3.6 #############

#
# This quickstart shows how to predict the intent of an utterance by using the LUIS REST APIs.
#

import requests

try:

    ##########
    # Values to modify.

    # YOUR-APP-ID: The App ID GUID found on the www.luis.ai Application Settings page.
    appId = ''

    # YOUR-PREDICTION-KEY: Your LUIS prediction key, 32 character value.
    prediction_key = ''

    # YOUR-PREDICTION-ENDPOINT: Replace with your prediction endpoint.
    # For example, "https://westus.api.cognitive.microsoft.com/"
    prediction_endpoint = 'https://westus.api.cognitive.microsoft.com/'

    # The utterance you want to use.
    meeting_text = ""
    # The headers to use in this REST call.
    headers = {
    }

    from spacy.lang.en import English

    nlp = English()
    nlp.add_pipe("sentencizer")
    doc = nlp(meeting_text)

    for utterance_span in doc.sents:
        utterance = ' '.join([token.text for token in utterance_span])
        print(utterance)

        # The URL parameters to use in this REST call.
        params = {
            'query': utterance,
            'timezoneOffset': '0',
            'verbose': 'true',
            'show-all-intents': 'true',
            'spellCheck': 'false',
            'staging': 'false',
            'subscription-key': prediction_key
        }


        # Make the REST call.
        response = requests.get(f'{prediction_endpoint}luis/prediction/v3.0/apps/{appId}/slots/production/predict', headers=headers, params=params)

        # Display the results on the console.
        intents = response.json()['prediction']['intents']
        top_intent_tuple = list(intents.items())[0]
        top_intent, score = top_intent_tuple
        score = score['score']
        if score > 0.95:
            print(top_intent)
        else:
            print('None')

except Exception as e:
    # Display the error string.
    print(f'{e}')