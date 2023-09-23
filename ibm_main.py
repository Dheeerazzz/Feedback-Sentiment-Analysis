#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[19]:


df=pd.read_csv('Restaurant_Reviews (1).tsv',delimiter='\t',quoting=3)


# In[20]:


df


# In[21]:


nltk.download('stopwords')


# In[23]:


corpus=[]
for i in range(0,1000):
    review=re.sub(pattern='[^a-zA-Z]',repl=' ',string=df['Review'][i])
    review=review.lower()
    review_words=review.split()
    review_words=[word for word in review_words if not word in set(stopwords.words('english'))]
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review_words]
    review=' '.join(review)
    corpus.append(review)


# In[28]:


from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=df.iloc[:,1].values


# In[29]:


from sklearn.model_selection import train_test_split
x_tr,x_test,y_tr,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# In[30]:


x_tr.shape,x_test.shape,y_tr.shape,y_test.shape


# In[31]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(x_tr,y_tr)


# In[32]:


y_pred=classifier.predict(x_test)
y_pred


# In[35]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

score1=accuracy_score(y_test,y_pred)
score2=precision_score(y_test,y_pred)
score3=recall_score(y_test,y_pred)

print("Accuracy score :",round(score1*100,2))
print("Precision score :",round(score2*100,2))
print("Recall score :",round(score3*100,2))


# In[36]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True,cmap='YlGnBu', xticklabels=['Negative','Positive'])
plt.xlabel("predicted values")
plt.ylabel("ctual values")


# In[40]:


best_accuracy=0.0
alpha_val=0.0
for i in np.arange(0.1,1.1,0.1):
    temp_classifier=MultinomialNB(alpha=i)
    temp_classifier.fit(x_tr,y_tr)
    tempy_pred=temp_classifier.predict(x_test)
    score=accuracy_score(y_test,tempy_pred)
    print("Accuracy score is", round(i,1),round(score*100,2))
    if score>best_accuracy:
        best_accuracy=score 
        alpha_val=i 
print("Best accuracy :",round(best_accuracy))


# In[41]:


classifier=MultinomialNB(alpha=0.2)
classifier.fit(x_tr,y_tr)


# In[42]:


def predict_sentiment(Sample_review):
    Sample_review=re.sub(pattern='[^a-zA-Z]',repl=' ',string=Sample_review)
    Sample_review=Sample_review.lower()
    Sample_review_words=Sample_review.split()
    Sample_review_words=[word for word in Sample_review_words if not word in set(stopwords.words('english'))]
    ps=PorterStemmer()
    final_review=[ps.stem(word) for word in Sample_review_words]
    final_review=' '.join(final_review)
    temp=cv.transform([final_review]).toarray()
    return classifier.predict(temp)


# In[61]:


sample="The food is horrible"
if predict_sentiment(sample):
    print('+')
else:
    print('-')


# In[63]:


sample="The desserts are ymumm"
if predict_sentiment(sample):
    print(predict_sentiment(sample))
    print('+')
else:
    print('-')


# In[64]:


sample="good food"
if predict_sentiment(sample):
    print('+')
else:
    print('-')


# In[ ]:




