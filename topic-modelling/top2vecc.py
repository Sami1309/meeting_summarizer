from top2vec import Top2Vec

from spacy.lang.en import English

text = ''' '''

nlp = English()
nlp.add_pipe("sentencizer")
doc = nlp(text)
sentences = []
for utterance_span in doc.sents:
    utterance = ' '.join([token.text for token in utterance_span])
    sentences.append(utterance)

# sentences *= 10
model = Top2Vec(sentences, min_count=1, embedding_model="universal-sentence-encoder-large")
# model.save()

model.load()
# model.hierarchical_topic_reduction()
topic_words, topic_scores, topic_nums = model.get_topics()

for words, scores, num in zip(topic_words, topic_scores, topic_nums):
    print(f'word in cluster {num}: {words}')
    # model.generate_topic_wordcloud(num)

_, _, _, topic_nums = model.search_topics(keywords=["agree"], num_topics=len(topic_nums))

from wordcloud import WordCloud
from scipy.special import softmax
import matplotlib.pyplot as plt

background_color="black"
for topic in topic_nums:
    # model.generate_topic_wordcloud(topic)
    word_score_dict = dict(zip(model.topic_words[topic],
                                       softmax(model.topic_word_scores[topic])))

    plt.figure(figsize=(16, 4),
            dpi=200)
    plt.axis("off")
    plt.imshow(
        WordCloud(width=1600,
                height=400,
                background_color=background_color).generate_from_frequencies(word_score_dict))
    plt.title("Topic " + str(topic), loc='left', fontsize=25, pad=20)

    plt.show()

documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=0, num_docs=10 )
for doc, score, doc_id in zip(documents, document_scores, document_ids):
    print(doc)
