# from chatbot import CB

import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model/model.h5')
import json
import random
intents = json.loads(open('dataset/data.json').read())
words = pickle.load(open('model/word.pkl','rb'))
classes = pickle.load(open('model/labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


from flask import Flask, render_template, request,url_for

app = Flask(__name__)

def index():
    return render_template('chatbot.html', image_url=url_for('static', filename='images/bg-bot.png'))

@app.route("/")
def splash():
    return render_template("landingpage/index.html")

@app.route("/home")
def home():
    return render_template("landingpage/home.html")

@app.route("/profile-desa")
def profile():
    return render_template("landingpage/profile.html")

@app.route("/peta-wilayah")
def peta():
    return render_template("landingpage/peta.html")

@app.route("/pemdes-sukajaya-berbenah-dengan-gerakan")
def berita1():
    return render_template("landingpage/berita1.html")
@app.route("/kegiatan-kader-pkk-desa-sukajaya")
def berita2():
    return render_template("landingpage/berita2.html")
@app.route("/penyampaian-visi-dan-misi-ketua-bumdes")
def berita3():
    return render_template("landingpage/berita3.html")
@app.route("/pembentukan-karang-taruna")
def berita4():
    return render_template("landingpage/berita5.html")

@app.route("/bencana-hujan-angin-yang-menimpa")
def berita5():
    return render_template("landingpage/berita4.html")

@app.route("/penetapan-pengurus-keanggotaan-rt-rw")
def berita6():
    return render_template("landingpage/berita6.html")
@app.route("/pengurusan-surat-keterangan-catatan-kepolisian")
def detail1():
    return render_template("landingpage/detail1.html")
@app.route("/pengurusan-surat-keterangan-tidak-mampu")
def detail2():
    return render_template("landingpage/detail2.html")
@app.route("/pengurusan-surat-keterangan-kehilangan")
def detail3():
    return render_template("landingpage/detail3.html")

@app.route("/pengurusan-surat-keterangan-melahirkan")
def detai4():
    return render_template("landingpage/detail4.html")

@app.route("/pengurusan-surat-keterangan-kematian")
def detai5():
    return render_template("landingpage/detail5.html")

@app.route("/surat-pengantar-pernikahan")
def detai6():
    return render_template("landingpage/detail6.html")

@app.route("/pengurusan-penduduk-domisili-usaha")
def detai7():
    return render_template("landingpage/detail7.html")

@app.route("/Pengurusan-Keterangan-Ijin-Usaha")
def detai8():
    return render_template("landingpage/detail8.html")

@app.route("/Pengurusan-Surat-keterangan-pindah")
def detai9():
    return render_template("landingpage/detail9.html")

@app.route("/Pengurusan-Surat-keterangan-bepergian")
def detai10():
    return render_template("landingpage/detail10.html")

@app.route("/Pengurusan-Kredit-Usaha-Rakyat")
def detai11():
    return render_template("landingpage/detail11.html")

@app.route("/surat-keterangan-permohonan-penurunan-atau-penaikan-daya-listrik")
def detai12():
    return render_template("landingpage/detail12.html")

@app.route("/permohonan-perubahan-nama")
def detai13():
    return render_template("landingpage/detail13.html")

@app.route("/Pembuatan-Akta-Kelahiran-anak")
def detai14():
    return render_template("landingpage/detail14.html")

@app.route("/Pembuatan-Kartu-Keluarga")
def detai15():
    return render_template("landingpage/detail15.html")

@app.route("/Pembuatan-Surat-Pengantar-BPJS-Kesehatan")
def detai16():
    return render_template("landingpage/detail16.html")

@app.route("/galeri-dan-foto")
def galeri1():
    return render_template("landingpage/galeri1.html")
@app.route("/galeri-dan-foto-2")
def galeri2():
    return render_template("landingpage/galeri2.html")
@app.route("/tentang-chatbot")
def tentang_chatbot():
    return render_template("landingpage/chatbot.html")




@app.route("/chatbot")
def chatbot():
    return render_template("chatbot/index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


app.run(debug = True)