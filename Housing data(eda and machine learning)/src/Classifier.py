import sqlite3
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report


con = sqlite3.connect("data/home_sales.db")
df = pd.read_sql_query("SELECT * FROM sales", con)
df_new=df.dropna()
df_new = df_new.reset_index(drop=True)
pd.options.mode.chained_assignment = None

class round_values(BaseEstimator,TransformerMixin):
    ''' this is a class to round values of bathrooms when called upon '''
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X['bathrooms']=X.bathrooms.apply(lambda x:round(x))
        return X
    


def change_price_to_binary(df):
    pricez = []
    for i in range(0,df.shape[0]):
        if df.loc[i,'price']>=75000 and df.loc[i,'price']<323000:
            a = 'low'
        elif df.loc[i,'price']>=323000 and df.loc[i,'price']<650000:
            a = 'medium'
        else:
            a = 'high'
        pricez.append(a)
    df_new['price'] = pricez

change_price_to_binary(df_new)

df_new['price']=pd.Categorical(df_new['price'],categories=['high','medium','low'],ordered=True)
le = LabelEncoder()
df_new['price']=le.fit_transform(df_new['price'])
#0 = 'high' , 1 ='low' , 2 ='medium'
y=df_new['price']
x=df_new[['living_room_size','latitude','bathrooms','longitude','lot_size','waterfront']]

model_RFC = make_pipeline( round_values(),
                          StandardScaler(),
                          RandomForestClassifier())


model_DTC = make_pipeline( round_values(),
                          StandardScaler(),
                          DecisionTreeClassifier())


#spliting to train and test data 

X_train,X_test,y_train,y_test = train_test_split(x , y ,test_size = 0.2,random_state=42)

# Train using pipeline 
model_RFC.fit(X_train,y_train)
model_DTC.fit(X_train,y_train)

#scoring and evluation 

pred = model_RFC.predict(X_test)
pred1 = model_DTC.predict(X_test)

# CV_Score 

CV_score_RFC = cross_val_score(model_RFC,x ,y ,cv=5,scoring='f1_micro')
CV_score_DTC = cross_val_score(model_DTC,x,y,cv=5 , scoring='f1_micro')

#plot tree


print('Random Forest Classifier was chosen as the optimal classifier as it had the highest F1 micro score compared to Decsision Tree Classifier.')
print('\n')
print('5 fold CV f1_micro Score for Random Forsest is', np.mean(CV_score_RFC))
print('\n')
print('5 fold CV f1_micro for Decision Tree is' , np.mean(CV_score_DTC))
print('\n')
print('Detailed report are as shown: ')
print('\n')
print('This is the classification report using RandomForestClassifier \n \n', classification_report(y_test,pred))
print('\n\nThis is the classification report using DecisionTreeClassifier \n \n', classification_report(y_test,pred1))
print('\n')
print('Although both regressor and classifier yielded similar prediction accuracy of ~80%, I am inclined to chose the classifier model. Base on my understanding, house pricing comes in ranges. As such, the classifier would be a more appropriate model to aid housing agents in gauging the price of a house. A regressor model would provide a single price point and would require prior experience or external inputs to aid housing agents in gauging house pricing.')

