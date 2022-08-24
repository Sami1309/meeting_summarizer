import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import spacy
from collections import defaultdict

prompt = ''''''

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    for text in texts:
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)

documents = [prompt]

from nltk.corpus import stopwords
stopwords = stopwords.words("english")

texts = [
    [word for word in document.lower().split() if word not in stopwords]
    for document in documents
]
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
# processed_text = gensim.utils.simple_preprocess(lemmatized_text, deacc=True)

lda_model = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)
print(lda_model.print_topics(num_topics=3))