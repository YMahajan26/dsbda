import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from collections import Counter
import math  # USED IN IDF FUNCTION

# Sample document

document = [
    "The fast and fast quick brown fox jumps over the lazy dog.A cat is sleeping peacefully on the sofa.The dog chases the cat around the garden."
]

# OR USE THIS DOCUMENT (THIS CONTAINS 3 SEPERATE DOCUMENTS)

# document = [
#     "The fast and fast quick brown fox jumps over the lazy dog.",
#     "A cat is fast sleeping peacefully on the sofa.",
#     "The dog chases the cat around the garden."
# ]


# Cleaning
punctuation = list(string.punctuation)
clean_documents = ["".join([char for char in doc if char not in punctuation]) for doc in document]

# Tokenization
tokens = [word_tokenize(doc.lower()) for doc in clean_documents]

# POS Tagging
pos_tags = [nltk.pos_tag(token) for token in tokens]

# Stop Words Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [[word for word in token if word.lower() not in stop_words] for token in tokens]

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [[stemmer.stem(word) for word in token] for token in filtered_tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [[lemmatizer.lemmatize(word) for word in token] for token in filtered_tokens]

# Print the results
print("--------------------------------------------------------------------")
print("Original Documents:")
print(document)
print("\nTokenization:")
print(tokens)
print("\nPOS Tagging:")
print(pos_tags)
print("\nStop Words Removal:")
print(filtered_tokens)
print("\nStemming:")
print(stemmed_tokens)
print("\nLemmatization:")
print(lemmatized_tokens)
print("-------------------------------------------------------------------------")

# -------------PART 2 OF QUESTION(TF and IDF)---------------------------------------

# Algorithm for Create representation of document by calculating TFIDF
# Function to calculate TF
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


# Function to calculate IDF
def computeIDF(documents):
    N = len(documents)
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
                for word, val in idfDict.items():
                    if val > 0:
                        idfDict[word] = math.log(N / float(val))
                    else:
                        idfDict[word] = 0
    return idfDict


# Function to calculate TF-IDF
def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


# Step 1: Import the necessary libraries.

# Step 2: Initialize the Documents.
documentA = 'cargo trucks are heavier than buses'
documentB = 'cars are faster than buses'

# Step 3: Create SPlit words for Document A and B. LIKE TOKENIZATION
list_wordA = documentA.split(' ')
list_wordB = documentB.split(' ')

# Step 4: Create Collection of Unique words from Document A and B.
uniqueWords = set(list_wordA ).union(set(list_wordB))

# Step 5: Create a dictionary of words and their occurrence for each document in the corpus
numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in list_wordA :
    numOfWordsA[word] += 1  # How many times each word is repeated
    numOfWordsB = dict.fromkeys(uniqueWords, 0)

for word in list_wordB:
    numOfWordsB[word] += 1

# Step 6: Compute the term frequency for each of our documents.
tfA = computeTF(numOfWordsA, list_wordA )
tfB = computeTF(numOfWordsB, list_wordB)

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

''' 
Cleaning = Removing punctuations from the text

stop_words = Words that doesnt hold much value or not so much important words like (I , my , me , myself, you, he , she etc....)

stemming = cutting or shortening of words ( "males" will be stemmed to "male" OR "playing" will become "play")

Lemmatization = Efficient way of reducing the word to its root form without changing the root maning (means it wont chnaging "tavelling" to "travel" as it would chanage the meaning of the line but), ( its can do "cities" to "city" as it wont chnag emuch of the maning and is still understandable.)
                it is advanced and efficient form of stemming

POS(part of speach) tagging =                           
    1)Nouns (NN): Words that represent people, places, things, or ideas.
    2)Verbs (VB): Words that express actions, events, or states of being.
    3)Adjectives (JJ): Words that describe or modify nouns.
    4)Adverbs (RB): Words that modify verbs, adjectives, or other adverbs to provide more information about time, place, manner, degree, etc.
    5)Pronouns (PRP): Words that can replace nouns or noun phrases.
    6)Prepositions (IN): Words that show the relationship between nouns or pronouns and other words in a sentence.
    7)Conjunctions (CC): Words that connect words, phrases, or clauses.
    8)Determiners (DT): Words that introduce nouns and clarify their reference.
    9)Interjections (UH): Words used to express strong emotions or sentiments.
    10)Particles (RP): Words that have grammatical function but don't fall into other categories.
    11)Numeral (CD): Words that represent numbers or numeric sequences.
    12)Punctuation (SYM): Marks used in writing to separate sentences, clauses, phrases, etc.


Term frequency = ratio or words kitni bar aye document mein
                TF = No. of occurance of a particular word / Total words in the document)


'''
