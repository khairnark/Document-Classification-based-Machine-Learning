import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import re

data = pd.read_csv("df_wihtout_stopwords.csv",encoding='latin-1')

#Printing all different types of categories
data.category.unique()

#converting category column into numeric target NUM_CATEGORY column
data['NUM_CATEGORY']=data.category.map({'Paystub':0,'SFHD':1,'PhotoPage':2})
data.tail()

#Splitting dataset into 60% training set and 40% test set
x_train, x_test, y_train, y_test = train_test_split(data.title, data.NUM_CATEGORY, random_state=50)

#Here we convert our dataset into a Bag Of Word model using a Bigram model
vect = CountVectorizer(ngram_range=(2,2))

#converting traning features into numeric vector
X_train = vect.fit_transform(x_train)

#converting training labels into numeric vector
X_test = vect.transform(x_test)

#Training and Predicting the data
mnb = MultinomialNB(alpha =0.2)
mnb.fit(X_train,y_train)
result= mnb.predict(X_test)
print(result)

#Printing accuracy of the our model
accuracy_score(result,y_test)

#This function return the class of the input news
def predict_news(news):
    test = vect.transform(news)
    pred= mnb.predict(test)
    if pred  == 0:
         return 'Paystub'
    elif pred == 1:
        return 'SFHD'
    elif pred == 2:
        return 'PhotoPage'
    else:
        return 'no class found'
        
#Copy and paste the news headline in 'x'
x=['PHOTOGRAPH ADDENDUM']
r = predict_news(x)
print (r)

# from sklearn import metrics
# from sklearn.metrics import accuracy_score
# import numpy as np

#Printing the confusion matrix of our prediction
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, result)


import joblib
joblib.dump(mnb, "document_classification2.pkl")
ml_classification = joblib.load('document_classification2.pkl')

test=['''DEPARTMENT OF HOMELAND SECURITY Federal Emergency Management Agency STANDARD FLOOD HAZARD DETERMINATION FORM (SFHDF) OMB Control No. 1660-0040 Expires: 10/31/18 SECTION | - LOAN INFORMATION 1. LENDER/SERVICER NAME AND ADDRESS. Customer Number 2. COLLATERAL DESCRIPTION (Building/Mobile Home/Property) (See instructions information.) Borrower: LIRA SGEH 1000200662 Address Determination Address: John serena John serena 28 street Road 28 street Road sanfrasco CT 12345 sanfrasco CT 12345 APN/Tax ID: 009-0004-017-0000 Lot: Block: S/D: Phase: Delivery Method: FDR-COM - WEB. Section: Township: Range: 3. LENDER/SERVICER ID # | 4. LOAN IDENTIFIER 5. AMOUNT OF FLOOD INSURANCE REQUIRED 50020000976 SECTION II IA. NATIONAL FLOOD INSURANCE PROGRAM (NFIP) COMMUNITY JURISDICTION 1. NFIP Community Name SACRAMENTO, CITY OF 2. County(ies) [e State {* NFIP Community Number SACRAMENTO COUNTY CA 060266 IB. NATIONAL FLOOD INSURANCE PROGRAM (NFIP) DATA AFFECTING BUILDING/MOBILE HOME 1. NFIP Map Number Community-Panel Number | 2. NFIP Map Panel Effective / 3. Is Letter Map Change (LOMC)? (Community name, "A") Revised Date @ No 06067C0190H August 16, 2012 . . CyYes _ (Ifyes, LOMC date/no. available, 4. Flood Zone 5. No NFIP Map enter date case no. below). x Date: Case Number: (Cc. FEDERAL FLOOD INSURANCE AVAILABILITY (Check apply.) 4. [X] Federal Flood Insurance available (community participates NFIP). Regular Program Emergency Program NFIP 2. Federal Flood Insurance available (community participate NFIP). Building/Mobile Home Coastal Barrier Resources Area (CBRA) Otherwise Protected Area (OPA). Federal Flood Insurance may available. CBRA/OPA Designation Date: ID. DETERMINATION IS BUILDING/MOBILE HOME IN SPECIAL FLOOD HAZARD AREA (ZONES CONTAINING THE LETTERS "A" OR"V")? [_] YES NO If yes, flood insurance required Flood Disaster Protection Act 1973. If no, flood insurance required Flood Disaster Protection Act 1973. Please note, risk flooding area reduced, removed. This determination based examining NFIP map, Federal Emergency Management Agency revisions it, information needed locate building /mobile home NFIP map. IE. COMMENTS (Optional) HMDA Information State: 06 County: 067 MSA/MD: 40900 cT: 0023.00 06067002300 LIFE OF LOAN DETERMINATION This flood determination provided solely use benefit entity named Section 1, Box 1 order comply 1994 Reform Act may used relied upon entity individual purpose, including, limited to, deciding whether purchase property determining value property. IF. PREPARER'S INFORMATION NAME, ADDRESS, TELEPHONE NUMBER (If Lender) DATE OF DETERMINATION John serena April 22, 2021 28 street Road sanfrasco CT 12345 Phone: ORDER 900.833.6347 Fax: NUMBER SFHDF - Form Page 1 1 FEMA Form 086-0-32 (06/16) Document created 04/23/2021 6:38.48 AM
''']
x = vect.transform(test)

print(ml_classification.predict(x))
