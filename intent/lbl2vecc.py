from lbl2vec import Lbl2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import models


todo_keywords = ["sure"] #["to do", "to-do", "make sure", "be sure to", "task", "job"]
question_keywords = ["?"] #["question", "ask", "how", "what", "why"]
decision_keywords = ["so"] #["align", "agree", "consensus", "decision", "decide", "plan"]

# iterable list of lists with descriptive keywords of type str. 
# For each label at least one descriptive keyword has to be added as list of str
descriptive_keywords = [todo_keywords, question_keywords, decision_keywords]

from spacy.lang.en import English

with open('/Users/keltonzhang/Desktop/Dell_Proj/STT/ref.txt') as f:
    meeting_text = f.read()
    nlp = English()
    nlp.add_pipe("sentencizer")
    doc = nlp(meeting_text)
    sentences = []
    for utterance_span in doc.sents:
        utterance = ' '.join([token.text for token in utterance_span])
        sentences.append(utterance)

    

# iterable list of gensim.models.doc2vec.TaggedDocument elements. 
# If you wish to train a new Doc2Vec model this parameter can not be None, whereas the doc2vec_model parameter must be None. 
# If you use a pretrained Doc2Vec model this parameter has to be None. 
# Input corpus, can be simply a list of elements, but for larger corpora, consider an iterable that streams the documents directly from disk/network
tagged_docs = [TaggedDocument(sent.split(' '), [i]) for i, sent in enumerate(sentences)]

#iterable list of custom names for each label. Label names and keywords of the same topic must have the same index.
label_names = ["to-do", "question", "decision"]  

# init model
model = Lbl2Vec(keywords_list=descriptive_keywords, tagged_documents=tagged_docs, min_count = 0, window=10000, label_names = label_names, doc2vec_model=None)
# model = models.Doc2Vec(tagged_docs, min_count = 2)
# train model which creates jointly embedded label, document and word vectors.
model.fit()

# model = models.Doc2Vec.load()

# vector = model.infer_vector('Just make sure you keep checking the company web site and the emails .'.split(' '))
# print(model.docvecs.most_similar([vector],topn=3))

similarity_scores = model.predict_model_docs()
assert(similarity_scores.shape[0]==len(sentences))

def print_top_k(k):
    top_k = similarity_scores.sort_values(by=['highest_similarity_score'], ascending=False).head(k)
    print(top_k)
    
    for idx in top_k["doc_key"]:
        print(sentences[idx])

print_top_k(3)