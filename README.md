```
meeting_summarizer
│   README.md
│   intent.py - Using azure cognitive services's speech sdk to detect custom intents from microphone input
|   luis.py - Using azure cognitive services's luis runtime to detect intents and entities from text
|   luis_api.py - Using luis REST api to classify intents of sentences
|   sdk_stt.py - Speech to text using azure cognitive services's speech sdk
|   transcribe_api.py - Speech to text using azure cognitive services's speech api
|   transcribe_sdk.py - Speech to text using azure cognitive services's speech sdk, takes audio from Azure blob storage, used in virtual machine setting 
│
└───STT
│   │   assemblyAI.py - Speech to text from file with AssemblyAI api
│   │   deepgram.py - Speech to text from file with Deepgram api
│   │   speech2text2.py - Speech to text from file with speech2text2 model
│   │   speech_transcript_with_hugging_face_🤗_transformers.py - Speech to text from file with wav2vec2 model
└───intent
    │   bert.py - unsupervised intent classification using pretrained BERT model
    │   intent_huggingface.py - unsupervised intent classification using finetuned T5 model
    │   lbl2vecc.py - unsupervised intent classification with custom keywords using lbl2vec
    │   coheree.py - supervised intent classification using Cohere's GPT model api
└───summary
    │   openai_summary_test.py - abstractive summarization using OpenAI's GPT model
    │   pegasus.py abstractive summarization using the pegasus-large model
    └───pegasus-large
        │   README.md
        │   config.json
        │   special_tokens_map.json
        │   spiece.model
        │   tokenizer_config.json
└───topic-modelling
    │   bertopicc.py - Generate topics with bertopic
    │   lda_gensim.py - Generate topics with Gensim's lda library
    │   openai_prompt.py - extract arbitrary number of text for arbitrary topic(e.g. extract 2 decisions from a paragraph) with OpenAI's GPT prompt api
    │   textrankk.py - Generate topics with textrank
    │   top2vecc.py - Generate topics with top2vec
```
