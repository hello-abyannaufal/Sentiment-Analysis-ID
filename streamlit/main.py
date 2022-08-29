import streamlit as st
import pandas as pd
import joblib
from Utility import TextPreprocessing
from Utility import DataPreprocessing

# Import dataset
df = pd.read_excel('../Instagram Cyber Bullying.xlsx', sheet_name='Sheet1')
# Drop unused feature
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.drop(['No.'], axis=1, inplace=True)

# Import model for testing
model = joblib.load(open('../SVM_Classifier.pkl', 'rb'))



# RENDER AREA
st.title('Sentiment Analysis ID Project')
st.write('By: Mohamad Abyannaufal')
st.write("I'm using this dataset to feed my machine so it can be smart. Please take a look! :D")
st.write("You can download dataset in here: https://www.kaggle.com/code/syauqiddjohan/skripsi-sentiment-analysis-project/data")

df

'''
### After Preprocessing
'''
# Train and Testing
comment_preprocessing = DataPreprocessing(df)
# Change value of df['Kategori'] to binary
comment_preprocessing.binarizer('Kategori', 'Non-bullying')
# Remove sign or punctuation in df['Komentar']
comment_preprocessing.text_preprocessing('Komentar')

df

'''
### Start Testing!
Start to typing so the machine can predict your language, don't forget to using Indonesian language
'''


text = st.text_input('Input your typing: ')
proc = TextPreprocessing(text, df)
final= TextPreprocessing(text, df)
if st.button('Start Classify!'):
    '''
    ### This is how machine work on your typing
    #### This is your original typing..
    '''
    st.write(text)
    '''
    #### Remove special sign on your typing..
    '''
    st.write(proc.remove_signs())
    '''
    #### Remove stopword on your typing..
    '''
    st.write(proc.remove_stopwords())
    '''
    #### Stemming and lowercasing! Back to your origins form!
    '''
    st.write(proc.text_stemming())
    '''
    #### Vectorized, be my machine food!
    '''
    st.write(proc.vectorizer(df))
    '''
    #### Your typing is classified:
    '''
    # Predicting Text
    result = model.predict(final.text_preprocessing())
    if result[0] == 0:
        '''
        ## Bullying
        '''
    else:
        '''
        ## Non-Bullying
        '''