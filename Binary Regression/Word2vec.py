import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# importing libraries
from nltk.tokenize import word_tokenize,sent_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,LancasterStemmer
#from contractions import fix
#from unidecode import unidecode
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV, StratifiedKFold
import pandas
from tqdm import tqdm
tqdm.pandas()
from wordcloud import WordCloud
from gensim.models import Word2Vec,doc2vec

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.svm import SVC,SVR

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity



from gensim.models import Word2Vec, KeyedVectors

path = "/Users/harryzzz/Documents/FakeNewsDet_Project/GoogleNews-vectors-negative300.bin"
word2vec_model = KeyedVectors.load_word2vec_format(path, binary=True)

word2vec_model

print(word2vec_model.vector_size)

print(word2vec_model['good'])

