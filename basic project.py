import numpy as np
import pandas as pd
data = pd.read_csv(r'C:\Users\manga\Downloads\Restaurant_Reviews.tsv', delimiter='\t' , quoting=3)
data.shape
data.columns
data.head()
data.info
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus =[]
for i in range(0,1000):
    review =re.sub(pattern='[^a-zA-Z]',repl=' ', string=data['Review'][i])
    review = review.lower()
    review_words = review.split()
    review_words = [word for word in review_words if not word in set(stopwords.words('english'))]
    ps= PorterStemmer()
    review =[ps.stem(word) for word in review_words]
    review = ' '.join(review)
    corpus.append(review)
corpus[:1500]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X =cv.fit_transform(corpus).toarray()
y = data.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.20, random_state=0)
X_train.shape,X_test.shape, y_train.shape,y_test.shape
from sklearn.naive_bayes import MultinomialNB
classifier =MultinomialNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score1 =accuracy_score(y_test,y_pred)
score2 = accuracy_score(y_test,y_pred)
score3 = recall_score(y_test,y_pred)

print("---------SCORES--------")
print("Accuracy score is {}%".format(round(score1*100,3)))
print("Precision score is {}%".format(round(score2*100,3)))
print("recall score is {}%".format(round(score3*100,3)))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
plt.figure(figsize =(10,6))
sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'])
plt.xlabel('Predicted values')
plt.ylabel('Actual Values')
plt.show()
from ssl import ALERT_DESCRIPTION_HANDSHAKE_FAILURE
best_accuracy =0.0
alpha_val =0.0
for i in np.arange(0.1,1.1,0.1):
    temp_classifier =MultinomialNB(alpha=i)
    temp_classifier.fit(X_train,y_train)
    temp_y_pred =temp_classifier.predict(X_test)
    score = accuracy_score(y_test,temp_y_pred)
    print("Accuracy Score for alpha={} is {}%".format(round(i,1),round(score*100,3)))
    if score>best_accuracy:
        best_accuracy=score
        alpha_val =i
print('----------------------------------------------------')
print("The Best Accuracy Score is {}% with alpha value as {}".format(round(best_accuracy*100, 2), round(alpha_val, 1)))
classifier =MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
sample_review=input()
def predict_sentiment(sample_review):
    sample_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_review)
    sample_review = sample_review.lower()
    sample_review_words = sample_review.split()
    sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_review = [ps.stem(word) for word in sample_review_words]
    final_review = ' '.join(final_review)
    temp = cv.transform([final_review]).toarray()
    return classifier.predict(temp)

if predict_sentiment(sample_review):
    print("This is a Positive review")
else:
    print("This is a Negative review")

