from flask import Flask, render_template, request
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as ENGLISH_STOP_WORDS
from spacy.lang.hi.stop_words import STOP_WORDS as HINDI_STOP_WORDS
from spacy.lang.kn.stop_words import STOP_WORDS as KANNADA_STOP_WORDS
from spacy.lang.ml.stop_words import STOP_WORDS as MALAYALAM_STOP_WORDS
from string import punctuation
from heapq import nlargest

app = Flask(__name__, template_folder='templates')

def summarizer(rawdocs, language="english"):
    if language == "english":
        stopwords = ENGLISH_STOP_WORDS
        nlp = spacy.load("en_core_web_sm")
    elif language == "hindi":
        stopwords = HINDI_STOP_WORDS
        nlp = spacy.load("xx_ent_wiki_sm")
    elif language == "kannada":
        stopwords = KANNADA_STOP_WORDS
        nlp = spacy.load("xx_ent_wiki_sm_kn")
    elif language == "malayalam":
        stopwords = MALAYALAM_STOP_WORDS
        nlp = spacy.load("xx_ent_wiki_sm_ml")
    else:
        raise ValueError("Unsupported language.")
    
    nlp.add_pipe('sentencizer')
    doc = nlp(rawdocs)
    tokens = [token.text for token in doc]
    word_freq={}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1
    max_freq=max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word]=word_freq[word]/max_freq
    sent_tokens=[sent for sent in doc.sents]
    sent_scores={}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent]=word_freq[word.text]
                else:
                    sent_scores[sent]+=word_freq[word.text]
    select_len=int(len(sent_tokens)*0.3)
    summary=nlargest(select_len,sent_scores,key=sent_scores.get)
    final_summary=[word.text for word in summary]
    summary=' '.join(final_summary)
    return summary, doc, len(rawdocs.split(' ')),len(summary.split(' '))
    # Rest of the summarizer function code goes here

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == "POST":
        rawtext = request.form['rawtext']
        language = request.form['language']
        
        supported_languages = ["english", "hindi", "kannada", "malayalam"]
        
        if language in supported_languages:
            try:
                summary, original_txt, len_orig_txt, len_summary = summarizer(rawtext, language)
                return render_template('summary.html', summary=summary, original_txt=original_txt, len_orig_txt=len_orig_txt, len_summary=len_summary)
            except ValueError as e:
                return render_template('error.html', message=str(e))
        else:
            return render_template('error.html', message="Unsupported language.")

if __name__ == "__main__":
    app.run(debug=True)
