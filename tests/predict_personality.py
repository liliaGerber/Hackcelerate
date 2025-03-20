import pickle
import re

import pandas as pd

C_EXT = pickle.load(open("data/models/cEXT.p", "rb"))
C_NEU = pickle.load(open("data/models/cNEU.p", "rb"))
C_AGR = pickle.load(open("data/models/cAGR.p", "rb"))
C_CON = pickle.load(open("data/models/cCON.p", "rb"))
C_OPN = pickle.load(open("data/models/cOPN.p", "rb"))
vectorizer_31 = pickle.load(open("data/models/vectorizer_31.p", "rb"))
vectorizer_30 = pickle.load(open("data/models/vectorizer_30.p", "rb"))


def predict_personality(text):
    sentences = re.split("(?<=[.!?]) +", text)
    text_vector_31 = vectorizer_31.transform(sentences)
    text_vector_30 = vectorizer_30.transform(sentences)
    ext = C_EXT.predict(text_vector_31)
    neu = C_NEU.predict(text_vector_30)
    agr = C_AGR.predict(text_vector_31)
    con = C_CON.predict(text_vector_31)
    opn = C_OPN.predict(text_vector_31)
    return [ext[0], neu[0], agr[0], con[0], opn[0]]


text = "It is important to note that each of the five personality "

predictions = predict_personality(text)
print("predicted personality:", predictions)
df = pd.DataFrame({"r": predictions, "theta": ["EXT", "NEU", "AGR", "CON", "OPN"]})
