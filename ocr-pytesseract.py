import numpy as np
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.datasets as skd
from csv import writer
import os
from bs4 import BeautifulSoup
import cv2
import pytesseract
import io
import nltk
import pandas as pd
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize


#Text Extraction
stop_words = set(stopwords.words('english'))
pytesseract.pytesseract.tesseract_cmd = 'C:/Users/username/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'
path = "C:/Users/username/train"
fields = ['Path', 'Extracted data']
filename = 'work_new4_testsetextracted.csv'
filelist = []
for root, dirs, files in os.walk(path):
    for file in files:
        #append the file name to the list
        filelist.append(os.path.join(root,file))
#         print(filelist)
with open(filename, 'a', newline='') as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(fields)
    for name in filelist:
        if '.png' in name:
            print(name)
            img = cv2.imread(name)
            exttext = pytesseract.image_to_string(img)
            words = exttext.split()
            appendFile = ''
            #newtxt = name + ".txt"
            #test_file = open(newtxt, "w")
            #test_file.write(exttext)
            #test_file.close()
            for r in words:
                if not r in stop_words:
                    appendFile = appendFile+' '+r
            rows = [
                name,
                appendFile,
            ]
            writer_object.writerow(rows)
 # # -------------------------------------------------------------------------------------           
def remove_punctuations(text):
     for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
        text = text.replace("\uFFFD", "\"")
        return text

stop_words = set(stopwords.words('english'))
path = filename
df = pd.read_csv(path)
df['without_stopwords'] = df['Extracted data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


df["new_column"] = df['without_stopwords'].apply(remove_punctuations)

# # -------------------------------------------------------------------------------------
def text_cleaning(a):
    remove_punctuation = [char for char in a if char not in string.punctuation]
    remove_punctuation = ''.join(remove_punctuation)
    return [word for word in remove_punctuation.split() if word.lower() not in stopwords.words('english')]

df['new_column2'] = df['new_column'].apply(text_cleaning)
df.to_csv('df_wihtout_stopwords.csv')            
