import sqlite3
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor


con = sqlite3.connect("data/home_sales.db")
df = pd.read_sql_query("SELECT * FROM sales", con)
df_new = df.dropna()
df_new = df_new.reset_index(drop=True)
x= df_new[['living_room_size','latitude','bathrooms','longitude','lot_size','waterfront']]
y = df_new['price']
pd.options.mode.chained_assignment = None

class round_values(BaseEstimator,TransformerMixin):
    ''' this is a class to round values of bathrooms when called upon '''
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        X['bathrooms']=X.bathrooms.apply(lambda x:round(x))
        return X
    

model_RFR = make_pipeline( round_values(),
                          StandardScaler(),
                          RandomForestRegressor())

#spliting to train and test data 

X_train,X_test,y_train,y_test = train_test_split(x , y ,test_size = 0.2,random_state=42)

# Train using pipeline 
model_RFR.fit(X_train,y_train)

#scoring and evluation 

pred = model_RFR.predict(X_test)

rsme = np.sqrt(mean_squared_error(y_test,pred))
r2= r2_score(y_test,pred)
print('The results for Random Forest Regressor are as follows: \n')
print("RandomForestRegressor RMSE: {:.2f} \n".format(rsme))
print("RandomForestRegressor r2 value: {:.5f} \n".format(r2))

compare = pd.DataFrame(
    {
     'Actual' : y_test,
     'predicited':pred,
     'Difference':y_test - pred
     }
    )

print('using Random Forest Regressor results : \n',compare.head())
print('\n')