import pandas as pd
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('kc_house_data.csv')
df = pd.DataFrame(df)
df['bathrooms'] = df['bathrooms'].round()
df1 = df.drop(columns=['id', 'date', 'lat', 'long'], axis='columns')

time = []
for i in df['yr_renovated']:
    if i == 0:
        time.append(0)
    else:
        a = 2015-int(i) #base year of the data is 2015
        time.append(a)
df1['years_since_last_renovation'] = time


built = []
for i in df['yr_built']:
    a = 2015 - int(i)
    built.append(a)
df1['age'] = built

x = df1.drop(['price', 'yr_renovated', 'yr_built', 'sqft_above', 'sqft_living15'], axis='columns')
y = df1['price']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=69)

rfr=RandomForestRegressor(n_estimators=100, random_state=69)
rfr.fit(x_train, y_train)

#Saving the model 
import joblib
filename = 'random_forest_regressor_model.pkl'
joblib.dump(rfr, filename)
print(f"Model successfully saved as '{filename}'")