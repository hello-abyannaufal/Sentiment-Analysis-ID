import pandas as pd
import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.Dictionary.ArrayDictionary import ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class TextPreprocessing:
    def __init__(self, text, dataframe):
        self.text = text
        self.dataframe = dataframe
    # THIS FUNCTION FOR PREPROCESSING TEXT IN DATAFRAME
    def remove_signs(self):
        # Remove number in string
        self.text = re.sub(r'[0-9]+', '', self.text)
        # Remove tab, new line, double space and back slice
        self.text = self.text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"").replace('\s+', " ")
        # Remove non ASCII (emoticon, chinese word, .etc)
        self.text = self.text.encode('ascii', 'replace').decode('ascii')
        # Remove mention, link, hashtag
        self.text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", self.text).split())
        # Remove incomplete URL
        self.text = self.text.replace("http://", " ").replace("https://", " ")
        # Remove doublespace and doubletick
        return self.text.replace('"', "").replace("'", "").replace("  ", " ")
    
    def remove_stopwords(self):
        factory = StopWordRemoverFactory()

        # You can custom stopwords list below, we will use stopword custom for remove stopword
        stopword_custom =["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 'jgn', 'sdh', 'aja', 'nyg', 'hehe', 'pen', 'nan', 'loh','&amp', 'yah']
        stopword_extend = pd.read_csv("../stopwordsID.csv")
        stopword_custom.extend(stopword_extend)

        # Add custom stopword to sastrawi and convert to dictionary
        stopword_sastrawi = factory.get_stop_words()+stopword_custom
        dictionary = ArrayDictionary(stopword_sastrawi)

        # Create StopWordRemover Function and add custom stopwords list
        stopword = StopWordRemover(dictionary)

        self.text = stopword.remove(self.text)
        return self.text

    def text_stemming(self):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        self.text = stemmer.stem(self.text)
        return self.text

    def vectorizer(self, dataframe) :
        X_train, X_test, y_train, y_test = train_test_split(dataframe['Komentar'], dataframe['Kategori'], shuffle=True, test_size=0.25, stratify=dataframe['Kategori'], random_state=30)
        # TF-IDF
        tfidf_vector = TfidfVectorizer(max_features=10000)
        tfidf_vector.fit_transform(X_train)
        output = tfidf_vector.transform([self.text])
        return output

    # Lazy Preprocessing
    def text_preprocessing(self):
        self.text = self.remove_signs()
        self.text = self.remove_stopwords()
        self.text = self.text_stemming()
        self.text = self.vectorizer(self.dataframe)
        return self.text

class DataPreprocessing:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    # THIS FUNCTION IS FOR MANIPULATING VALUE OF FEATURE
    def binarizer(self, feature, positive):
        def toBinary(text, positive):
            if text == positive:
                text = 1
            else:
                text = 0
            return text

        sentence = []
        for index, row in self.dataframe.iterrows():
            sentence.append(toBinary(row[feature], positive))

        self.dataframe[feature] = sentence

    # THIS FUNCTION FOR PREPROCESSING TEXT IN DATAFRAME
    def remove_signs(self, feature):
        def delSign(text):
            # Remove number in string
            text = re.sub(r'[0-9]+', '', text)
            # Remove tab, new line, double space and back slice
            text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"").replace('\s+', " ")
            # Remove non ASCII (emoticon, chinese word, .etc)
            text = text.encode('ascii', 'replace').decode('ascii')
            # Remove mention, link, hashtag
            text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
            # Remove incomplete URL
            text = text.replace("http://", " ").replace("https://", " ")
            # Remove doublespace and doubletick
            return text.replace('"', "").replace("'", "").replace("  ", " ")

        sentence = []
        for index, row in self.dataframe.iterrows():
            sentence.append(delSign(row[feature]))
        
        self.dataframe[feature] = sentence
    
    def remove_stopwords(self, feature):
        factory = StopWordRemoverFactory()

        # You can custom stopwords list below, we will use stopword custom for remove stopword
        stopword_custom =["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang', 'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'jd', 'jgn', 'sdh', 'aja', 'nyg', 'hehe', 'pen', 'nan', 'loh','&amp', 'yah']
        stopword_extend = pd.read_csv("../stopwordsID.csv")
        stopword_custom.extend(stopword_extend)

        # Add custom stopword to sastrawi and convert to dictionary
        stopword_sastrawi = factory.get_stop_words()+stopword_custom
        dictionary = ArrayDictionary(stopword_sastrawi)

        # Create StopWordRemover Function and add custom stopwords list
        stopword = StopWordRemover(dictionary)

        sentence = []
        for index, row in self.dataframe.iterrows():
            sentence.append(stopword.remove(row[feature]))

        self.dataframe[feature] = sentence

    def text_stemming(self, feature):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        def stemming(text):
            text = stemmer.stem(text)
            return text
        
        sentence = []
        for index, row in self.dataframe.iterrows():
            sentence.append(stemming(row[feature]))

        self.dataframe[feature] = sentence

    
    # Lazy Preprocessing
    def text_preprocessing(self, feature):
        self.remove_signs(feature)
        self.remove_stopwords(feature)
        self.text_stemming(feature)
