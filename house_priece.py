# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 14:09:48 2018

@author: LMC
"""
# bibliotecas usadas
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#importando os arquivos
main_file_path = 'train_house.csv'
data = pd.read_csv(main_file_path)
data.describe()
test = pd.read_csv('test_house.csv')

print(data.columns)

price_data = data.SalePrice
print(price_data.head())

columns_of_interest = ['SaleCondition', 'HouseStyle']
two_columns_of_data = data[columns_of_interest]

two_columns_of_data.describe()

y = data.SalePrice
melbourne_predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = data[melbourne_predictors]
test_X = test[melbourne_predictors]

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

iowa_model = RandomForestRegressor()
iowa_model.fit(train_X, train_y)
iowa_model.fit(train_X, train_y)

predicted_prices = iowa_model.predict(test_X)

print(predicted_prices)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(iowa_model.predict(X.head()))

predicted_home_prices = iowa_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

val_predictions = iowa_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)
