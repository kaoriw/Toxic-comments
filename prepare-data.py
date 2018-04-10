import re
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
train_small = pd.read_csv('train.csv',nrows=500)
test = pd.read_csv('test.csv')
test_small = pd.read_csv('test.csv',nrows=500)
sample_submission_small = pd.read_csv('sample_submission.csv',nrows=500)

# generate train_small.csv, test_small.csv and sample_submission_small.csv
train_small.to_csv('train_small.csv',header=True,index=True)
test_small.to_csv('test_small.csv',header=True,index=True)
sample_submission_small.to_csv('sample_submission_small.csv',header=True,index=True)

##############################
##       clean data         ##
##############################
#Regex to remove all Non-Alpha Numeric and space
special_character_removal=re.compile(r'[^a-z\d ]',re.IGNORECASE)
#regex to replace all numerics
replace_numbers=re.compile(r'\d+',re.IGNORECASE)
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.    
    # Convert words to lower case and split them
    text = text.lower().split()
    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]    
    text = " ".join(text)   
    #Remove Special Characters
    text=special_character_removal.sub('',text)    
    #Replace Numbers
    text=replace_numbers.sub('n',text)   
    # Return a list of words
    return(text)

sentences_train = train["comment_text"].fillna("NA").values
sentences_test = train["comment_text"].fillna("NA").values
comments_train = []
for text1 in sentences_train:
    comments_train.append(text_to_wordlist(text1))
comments_test = []
for text2 in sentences_test:
    comments_test.append(text_to_wordlist(text2))
comments_all = comments_train + comments_test
comments_all = pd.DataFrame(data=comments_all)
# generate comments_all.csv, use this file to do GloVe
comments_all.to_csv('comments_all.csv') 

