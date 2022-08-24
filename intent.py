# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

import time
import azure.cognitiveservices.speech as speechsdk

"""performs one-shot intent recognition from input from the default microphone"""

# <create_speech_configuration>
# Set up the config for the intent recognizer (remember that this uses the Language Understanding key, not the Speech Services key)!
intent_config = speechsdk.SpeechConfig(
    subscription="",
    region="")
# </create_speech_configuration>

# <create_intent_recognizer>
# Set up the intent recognizer
audio_config = speechsdk.AudioConfig(filename="xx.wav")

intent_recognizer = speechsdk.intent.IntentRecognizer(speech_config=intent_config, audio_config=audio_config)
# </create_intent_recognizer>

# <add_intents>
# set up the intents that are to be recognized. These can be a mix of simple phrases and
# intents specified through a LanguageUnderstanding Model.
model = speechsdk.intent.LanguageUnderstandingModel(app_id="")
# intents = [
#     (model, "HomeAutomation.TurnOn"),
#     (model, "HomeAutomation.TurnOff"),
#     ("This is a test.", "test"),
#     ("Switch to channel 34.", "34"),
#     ("what's the weather like", "weather"),
# ]
# intent_recognizer.add_intents(intents)
# </add_intents>

# To add all of the possible intents from a LUIS model to the recognizer, uncomment the line below:
intent_recognizer.add_all_intents(model);


# Starts intent recognition, and returns after a single utterance is recognized. The end of a
# single utterance is determined by listening for silence at the end or until a maximum of 15
# seconds of audio is processed. It returns the recognition text as result.
# Note: Since recognize_once() returns only a single utterance, it is suitable only for single
# shot recognition like command or query.
# For long-running multi-utterance recognition, use start_continuous_recognition() instead.
# <recognize_intent>
'''
intent_result = intent_recognizer.start_continuous_recognition(audio_input)
# </recognize_intent>



# <print_results>
# Check the results
if intent_result.reason == speechsdk.ResultReason.RecognizedIntent:
    print("Recognized: \"{}\" with intent id `{}`".format(intent_result.text, intent_result.intent_id))
elif intent_result.reason == speechsdk.ResultReason.RecognizedSpeech:
    print("Recognized: {}".format(intent_result.text))
elif intent_result.reason == speechsdk.ResultReason.NoMatch:
    print("No speech could be recognized: {}".format(intent_result.no_match_details))
elif intent_result.reason == speechsdk.ResultReason.Canceled:
    print("Intent recognition canceled: {}".format(intent_result.cancellation_details.reason))
    if intent_result.cancellation_details.reason == speechsdk.CancellationReason.Error:
        print("Error details: {}".format(intent_result.cancellation_details.error_details))
# </print_results>'''

def recognize_intent_continuous():
    """performs continuous intent recognition from input from an audio file"""
    # <IntentContinuousRecognitionWithFile>

    # Set up the intent recognizer
    intent_recognizer = speechsdk.intent.IntentRecognizer(speech_config=intent_config, audio_config=audio_config)

    # set up the intents that are to be recognized. These can be a mix of simple phrases and
    # intents specified through a LanguageUnderstanding Model.

    # Connect callback functions to the signals the intent recognizer fires.
    done = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    intent_recognizer.session_started.connect(lambda evt: print("SESSION_START: {}".format(evt)))
    intent_recognizer.speech_end_detected.connect(lambda evt: print("SPEECH_END_DETECTED: {}".format(evt)))
    # event for intermediate results
    intent_recognizer.recognizing.connect(lambda evt: print("RECOGNIZING: {}".format(evt)))
    # event for final result
    intent_recognizer.recognized.connect(lambda evt: print(
        "RECOGNIZED: {}\n\tText: {} (Reason: {})\n\tIntent Id: {}\n\tIntent JSON: {}".format(
            evt, evt.result.text, evt.result.reason, evt.result.intent_id, evt.result.intent_json)))

    # cancellation event
    intent_recognizer.canceled.connect(lambda evt: print(f"CANCELED: {evt.cancellation_details} ({evt.reason})"))

    # stop continuous recognition on session stopped, end of speech or canceled events
    intent_recognizer.session_stopped.connect(stop_cb)
    intent_recognizer.speech_end_detected.connect(stop_cb)
    intent_recognizer.canceled.connect(stop_cb)

    # And finally run the intent recognizer. The output of the callbacks should be printed to the console.
    intent_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    intent_recognizer.stop_continuous_recognition()

recognize_intent_continuous()