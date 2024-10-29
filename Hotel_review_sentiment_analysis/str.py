import pandas as pd
from nltk.corpus import wordnet
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import WhitespaceTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve,auc

df = pd.read_csv(r"C:\Users\jayas\Downloads\Hotel_Reviews.csv")

df.head()
df["Review"] = df["Positive_Review"] + df["Negative_Review"]
df["Is_bad_review"] = df["Reviewer_Score"].apply(lambda x : 1 if x > 5 else 0)
df = df[["Review","Is_bad_review"]]
df = df.sample(frac=0.1,random_state=0,replace=False)
df["Review"] = df["Review"].apply(lambda x: x.replace("No Negative","").replace("No Positive",""))

def get_word_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    text = text.lower()
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    text = [word for word in text if not any(c.isdigit() for c in word)]
    stopwords1 = stopwords.words("english")
    text = [x for x in text if x not in stopwords1]
    text = [t for t in text if len(t) > 0]
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_word_pos(t[1])) for t in pos_tags]
    text = [t for t in text if len(t) > 1]
    text = " ".join(text)
    return text
df['clean_reviews'] = df["Review"].apply(lambda x: clean_text(x))

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sent = SentimentIntensityAnalyzer()
df["Sentiments"] = df["Review"].apply(lambda x : sent.polarity_scores(x))
df = pd.concat([df.drop(["Sentiments"],axis = 1),df["Sentiments"].apply(pd.Series)],axis = 1)

df["nb_chars"] = df["Review"].apply(lambda x : len(x))
df["nb_words"] = df["Review"].apply(lambda x: len(x.split(" ")))

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df["Review"].apply(lambda x : x.split(" ")))]
model = Doc2Vec(documents,vector_size=5,window=2,min_count = 1,workers = 4)
doc2_vex_df = df["clean_reviews"].apply(lambda x : model.infer_vector(x.split(" "))).apply(pd.Series)
doc2_vex_df.columns = ["doc2vec_vector" + str(x) for x in doc2_vex_df.columns]
df = pd.concat([df,doc2_vex_df],axis = 1)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df = 10)
tfidf_res = tfidf.fit_transform(df["clean_reviews"]).toarray()
tfidf_df = pd.DataFrame(tfidf_res,columns=tfidf.get_feature_names_out())
tfidf_col = ["word" + str(x) for x in tfidf_df.columns]
tfidf_df.index = df.index
df = pd.concat([df,tfidf_df],axis= 1)

df["Is_bad_review"].value_counts(normalize = True)

wordcloud = WordCloud(max_words = 200,scale = 5,random_state = 42,background_color = 'white').generate(str(df["Review"]))
plt.figure(1,figsize = (20,20))
plt.imshow(wordcloud)
df[df["nb_words"] >= 5].sort_values("pos",ascending = False)[["Review","pos"]].head(10)

for x in [0,1]:
    subset = df[df["Is_bad_review"] == x]
    if x == 0:
        label = "Good Reviews"
    else:
        label = "Bad Reviews"
    sns.distplot(subset['compound'],hist = False,label = label)

ignore_cols = ["Is_bad_review","Review","clean_reviews"]
features = [c for c in df.columns if c not in ignore_cols]

x_train,x_test,y_train,y_test = train_test_split(df[features],df["Is_bad_review"],test_size=0.20,random_state= 0)
rf = RandomForestClassifier(n_estimators= 100,random_state=0)
rf.fit(x_train,y_train)
feature_df = pd.DataFrame({"features" : features,"importance" : rf.feature_importances_}).sort_values("importance",ascending=False)
feature_df.head()

y_pred = [x[1] for x in rf.predict_proba(x_test)]
fpr,tpr,thresolds = roc_curve(y_test,y_pred,pos_label=1)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label = "ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()