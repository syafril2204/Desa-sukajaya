from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer
# Creating ChatBot Instance
CB = ChatBot('ChatBot')
 # Training with Personal Ques & Ans
conversation = [
    "Hello",
    "halo kamu",
    "siapa pembuatmu?",
    "manusia",
    "siapa kamu?",
    "Saya Chatbot",
    "NKRI adalah?",
    "Negara Kesatuan Republik Indonesia",
    "tahun berapa indonesia merdeka?",
    "1945",
    "sekarang jam berapa?",
    "21.06"
]
trainer = ListTrainer(CB)
trainer.train(conversation)
# # Training with English Corpus Data
trainer_corpus = ChatterBotCorpusTrainer(CB)
trainer_corpus.train('chatterbot.corpus.english')
