import gc
import re
import nltk
import string
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from distributed import LocalCluster
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import WordPunctTokenizer, RegexpTokenizer, TreebankWordTokenizer

STOP=set(stopwords.words('english') + list(string.punctuation))

# cluster = LocalCluster()

def plot_confusion_matrix(cm, target_names,title='Confusion matrix',cmap=None, normalize=True):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

# function to parse the titles for html data
def parse_title(x):
    soup = BeautifulSoup(x,'html.parser')
    val = soup.find('title')
    if val is not None:
        val = val.text
        val = val.strip()
        return val
    else:
        return ''

# function to write titles to a csv file
def extract_text(file,fn,outfile,chunk_size=10**3):
    for chunk in pd.read_csv(file, chunksize=chunk_size, encoding='utf-8', engine='c'):
        df = chunk['Webpage_id']
        title = chunk['Html'].apply(lambda x: fn(x))
        df = pd.concat([df,title], axis=1)
        df.to_csv(outfile, mode='a',encoding='utf-8',index=False)
        del df,title

# function to preprocess the text data
def process_data(df, col):
    df[col]=df[col].apply(lambda x: re.sub(r'www|http',"", str(x)))
    df[col]=df[col].apply(lambda x: re.sub(r'\d+',"", str(x)))
    df[col]=df[col].apply(lambda x: re.sub(r'-'," ", str(x)))

    return df[col]

# custom tokenizer
def tokenize(text):
    tokens = [word for word in wpt.tokenize(text) if len(word) > 1]
    stems = [stemmer.stem(item) for item in tokens]
    
    return stems

# function to prepare submission file
def prepare_sub(prediction, filename):
    sub = pd.read_csv('./data/sample_submission_poy1UIu.csv')
    sub.Tag = prediction
    sub.to_csv(filename, index=False)

# extract the titles. This will take some time
# extract_text('./data/train/html_data.csv',parse_title, 'w_id_title.csv', chunk_size=10**4)

labels=['news', 'clinicalTrials', 'conferences', 'profile', 'forum','publication', 'thesis', 'guidelines', 'others']

tr = pd.read_csv('./data/train/train.csv')
ts = pd.read_csv('./data/test_nvPHrOx.csv')

titles = pd.read_csv(open('w_id_title.csv','rU'), encoding='utf-8', engine='c')
titles.columns = ['Webpage_id', 'Title']
tr.Webpage_id=tr.Webpage_id.astype('str')
ts.Webpage_id=ts.Webpage_id.astype('str')

# merge the titles with train and test files
tr = pd.merge(tr,titles)
ts = pd.merge(ts, titles)

# tr['full_text']
tr['full_text']=tr.Domain + ' ' + tr.Url + ' ' + tr.Title
ts['full_text']=ts.Domain + ' ' + ts.Url + ' ' + ts.Title

# only domain field
tr_dom= process_data(tr,'full_text')
ts_dom= process_data(ts,'full_text')


y=tr.loc[:,'Tag']
X=tr_dom

del tr, ts
gc.collect()

stemmer = SnowballStemmer("english", ignore_stopwords=True)
wpt=RegexpTokenizer(r'\w+')
# wpt=WordPunctTokenizer()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# SGD Classifier
text_clf_svm = Pipeline([('vect', TfidfVectorizer(stop_words=STOP, ngram_range=(1,3), analyzer='word', min_df=0.003)),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=10, random_state=42))])

text_clf_svm_w = Pipeline([('vect', TfidfVectorizer(stop_words=STOP, ngram_range=(1,3), analyzer='char', min_df=0.003)),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=10, random_state=23))])

text_clf_svm_wn = Pipeline([('vect', TfidfVectorizer(stop_words=STOP, ngram_range=(1,3), analyzer=tokenize, min_df=0.003)),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=10, random_state=90))])

# Multinomial Naive bayes
text_clf_mnb = Pipeline([('vect', TfidfVectorizer(stop_words=STOP, ngram_range=(1,3), analyzer='word', min_df=0.003)),
                         ('clf-nb', MultinomialNB(alpha=1e-2,fit_prior=True, class_prior=None))])

text_clf_mnb_w = Pipeline([('vect', TfidfVectorizer(stop_words=STOP, ngram_range=(1,3), analyzer='char', min_df=0.003)),
                         ('clf-nb', MultinomialNB(alpha=1e-2,fit_prior=True, class_prior=None))])

text_clf_mnb_wn = Pipeline([('vect', TfidfVectorizer(stop_words=STOP, ngram_range=(1,3), analyzer=tokenize, min_df=0.003)),
                         ('clf-nb', MultinomialNB(alpha=1e-2,fit_prior=True, class_prior=None))])


# LinearSVC model
text_clf_svc = Pipeline([('vect', TfidfVectorizer(stop_words=STOP, ngram_range=(1,3), analyzer='word', min_df=0.003)),
                         ('clf-svc', LinearSVC(C=1e-2,class_weight='balanced',random_state=84))])

text_clf_svc_w = Pipeline([('vect', TfidfVectorizer(stop_words=STOP, ngram_range=(1,3), analyzer='char', min_df=0.003)),
                         ('clf-svc', LinearSVC(C=1e-2,class_weight='balanced',random_state=102))])

text_clf_svc_wn = Pipeline([('vect', TfidfVectorizer(stop_words=STOP, ngram_range=(1,3), analyzer=tokenize, min_df=0.003)),
                         ('clf-svc', LinearSVC(C=1e-2,class_weight='balanced',random_state=101))])

# logistic Regression
text_clf_log = Pipeline([('vect', TfidfVectorizer(stop_words=STOP, ngram_range=(1,3), analyzer='char', min_df=0.003)),
                        #  ('scaler',StandardScaler()),
                         ('clf-log', LogisticRegression(random_state=200, multi_class='multinomial', solver='newton-cg'))])


estm = [('sgd', text_clf_svm),('sgd_w', text_clf_svm_w), ('sgd_wn', text_clf_svm_wn), 
        ('mnb', text_clf_mnb), ('mnb_w', text_clf_mnb_w), ('mnb_wn', text_clf_mnb_wn), 
        ('svc', text_clf_svc), ('svc_w', text_clf_svc_w), ('svc_wn', text_clf_svc_wn),
        ('log', text_clf_log)]


clfens = VotingClassifier(estimators=estm, voting='hard')

mdlfit = clfens.fit(tr_dom.values, y.values)
pred = mdlfit.predict(ts_dom)
prepare_sub(pred,'ensemble_tfidf_svd.csv')
