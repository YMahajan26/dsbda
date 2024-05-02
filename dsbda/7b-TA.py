
import pandas as pd
import nltk #natural language tool kit library widely used for NLP applications
import re # regular expression used for pattern matching
# nltk.download('punkt')
# nltk.download('stopwords')
nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
#------------------------------------------------------------------------------------------------------------------------------
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict
#------------------------------------------------------------------------------------------------------------------------------
def computeIDF(documents):
    import math
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
                for word, val in idfDict.items():
                    if val > 0: idfDict[word] = math.log(N / float(val))
                    else: idfDict[word] = 0

    return idfDict
#------------------------------------------------------------------------------------------------------------------------------
def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

text= "Tokenization is the first step in text analytics. The process of breaking down a text paragraph into smaller chunks such as words or sentences is called Tokenization."
print('The given sentences are: \n', text)
#------------------------------------------------------------------------------------------------------------------------------
#Sentence Tokenization
from nltk.tokenize import sent_tokenize
tokenized_text= sent_tokenize(text)
print("\n Sentence Tokenization: \n", tokenized_text)
#Word Tokenization

from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print('\nWord Tokeniztion: \n', tokenized_word)
#------------------------------------------------------------------------------------------------------------------------------
# Add code for POS Tagging
#------------------------------------------------------------------------------------------------------------------------------
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
# Removing stop words
text= "How to remove stop words with NLTK library in Python?"
text= re.sub('[^a-zA-Z]', ' ',text)
tokens = word_tokenize(text.lower())
filtered_text=[]
for w in tokens:
    if w not in stop_words:
        filtered_text.append(w)
print ("Tokenized Sentence:",tokens)
print ("Filterd Sentence:",filtered_text)
#------------------------------------------------------------------------------------------------------------------------------
#Stamming
from nltk.stem import PorterStemmer
e_words= ["wait", "waiting", "waited", "waits"]
ps =PorterStemmer()
for w in e_words:
    rootWord=ps.stem(w)
    print('Stemming for ',w,': ',rootWord)
#Lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
text = "studies studying cries cry"
tokenization = nltk.word_tokenize(text)
for w in tokenization:
    print("Lemma for {} is {}".format(w,
    wordnet_lemmatizer.lemmatize(w)))
#------------------------------------------------------------------------------------------------------------------------------
# Algorithm for Create representation of document by calculating TFIDF
# Step 1: Import the necessary libraries.
from sklearn.feature_extraction.text import TfidfVectorizer
# Step 2: Initialize the Documents.
documentA = 'Jupiter is the largest planet'
documentB = 'Mars is the fourth planet from the Sun'
# Step 3: Create BagofWords (BoW) for Document A and B. word tokenization
bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')
# Step 4: Create Collection of Unique words from Document A and B.
uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))

# Step 5: Create a dictionary of words and their occurrence for each document in the corpus
numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1 #How many times each word is repeated
numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] += 1
# Step 6: Compute the term frequency for each of our documents.
tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)
# Step 7: Compute the term Inverse Document Frequency.
print('----------------Term Frequency----------------------')
df = pd.DataFrame([tfA, tfB])
print(df)
# Step 8: Compute the term TF/IDF for all words.
idfs = computeIDF([numOfWordsA, numOfWordsB])
print('----------------Inverse Document Frequency----------------------')
print(idfs)
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
print('------------------- TF-IDF--------------------------------------')
df = pd.DataFrame([tfidfA, tfidfB])
print(df)