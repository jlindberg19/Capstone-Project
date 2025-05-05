# -*- coding: utf-8 -*-
"""
Created on Sun May  4 23:09:22 2025

@author: Jacob Lindberg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb
from sklearn import metrics
import datetime as dt

# Function that gives date features from a date time value
def date_features(df, label=None):
    df['quarter'] = df['WK'].dt.quarter
    df['month'] = df['WK'].dt.month
    df['year'] = df['WK'].dt.year
    df['week'] = df['WK'].dt.isocalendar().week
    
    X = df[['quarter','month','year',
           'week']]
    if label:
        y = df[label]
        return X, y
    return X


# Data Initialization
sales_data = pd.read_csv('C:/Users/a/Documents/Vendor_Sales_Data.csv')

sales_data['Out of Stock %'] = sales_data['Out of Stock %'].astype(float)
sales_data['WK'] = pd.to_datetime(sales_data['WK'], errors ='coerce')


date_features(sales_data)

new_asp = []
for row in range(len(sales_data)):
    if pd.isna(sales_data.iloc[row]['ASP']) == True:
        new_asp.append(float(sales_data.iloc[row]['Reg Retail']))
    else:
        new_asp.append(float(sales_data.iloc[row]['ASP'])*.98)
sales_data['ASP'] = new_asp

sales_data['Price %'] = sales_data['ASP']/sales_data['Reg Retail']
sales_data.fillna({'Price %':.98}, inplace=True)

zeroSales = sales_data[ (sales_data['Sales Units'] == 0)].index
sales_data.drop(zeroSales, inplace=True)



unencoded_sales_data = sales_data

skus1 = unencoded_sales_data['SKU'].unique().tolist()
# Encoding qualitative variables 
encoder = OneHotEncoder(sparse_output=False)
cat_columns = ['SKU', 'Primary Category', 'PCM Product Definition']
one_hot_encoded = encoder.fit_transform(sales_data[cat_columns])
oneHot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(cat_columns))


sales_data = pd.concat([sales_data, oneHot_df], axis=1)
sales_data = sales_data.drop(cat_columns, axis=1)
sales_data = sales_data.dropna()

# Model Creation


X = sales_data.drop(columns=['Sales Units', 'WK', 'DESC', 'ASP', 'Reg Retail'], axis=1)
Y = sales_data['Sales Units']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

regressor = xgb.XGBRegressor()
reg_cv = GridSearchCV(regressor, {"colsample_bytree":[1.0], "min_child_weight": [1.0, 1.2, 1.4, 1.6], "max_depth":[3, 4, 6, 7], 'n_estimators':[500, 1000, 1500]}, verbose =1)
reg_cv.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)], verbose=False)

regressor = xgb.XGBRegressor(**reg_cv.best_params_)
regressor.fit(X_train, Y_train)

predictions = regressor.predict(X_test)
print(regressor.score(X_test, Y_test))
regressor.score(X_train, Y_train)

#

results = pd.DataFrame()
results['features'] = X.columns
results['importances'] = regressor.feature_importances_

# Anonymizing Data at request of employer

new_names = []
for r in range(len(results)):
    if 'SKU_' in results.loc[r]['features']:
        new_names.append('SKU_'+str(r))
    elif 'Primary Category_' in results.loc[r]['features']:
        new_names.append('PC_'+str(r))
    elif 'PCM Product Definition_' in results.loc[r]['features']:
        new_names.append('Definition_'+str(r))
    else:
        new_names.append(results.loc[r]['features'])

anon_results = results
anon_results['features'] = new_names

anon_results.sort_values(by = 'importances', ascending = False, inplace = True)
anon_results[:20]


import matplotlib.pyplot as plt

plt.style.use('_mpl-gallery')

x = results[:10]['features']
y = results[:10]['importances']

fig, ax = plt.subplots()
ax.barh(x, y, edgecolor="white")
plt.show()

# Sales Data vs Predictions

plotSKU = 1
weeksFwd = 10
weeksBkwd = 10
strCnt = unencoded_sales_data[( unencoded_sales_data['SKU'] == plotSKU)]['Store Count'].tail(1)

historic_sales = unencoded_sales_data[( unencoded_sales_data['SKU'] == plotSKU)]['Sales Units'].tail(weeksBkwd)
outStockP = unencoded_sales_data[( unencoded_sales_data['SKU'] == plotSKU)]['Out of Stock %'].tail(weeksBkwd).mean()
priceP = unencoded_sales_data[( unencoded_sales_data['SKU'] == plotSKU)]['Price %'].tail(weeksBkwd).mean()
predSales = []
def PredictedSales(df, sku, week_start, week_end, storeCount):
    total_sales = 0
    year = 2025
    input_data = df.loc[( df['SKU_'+str(sku)] == 1)].iloc[-1]
    input_data['Price %'] = priceP
    input_data['Store Count'] = storeCount
    input_data['Out of Stock %'] = outStockP
    input_data['Out of Stock Count'] = int(storeCount*outStockP)
    for w in range(week_end - week_start):
        if week_start + w > 52:
            week_start -= 52
            year += 1
        input_data['week'] = week_start + w
        input_data['month'] = dt.date.fromisocalendar(year, week_start + w, 7).month
        input_data['quarter'] = pd.Timestamp(dt.date.fromisocalendar(year, week_start + w, 7)).quarter      
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = regressor.predict(input_data_reshaped)
        total_sales = total_sales + prediction
    return int(total_sales[0])

for w in range(weeksFwd):
    predSales.append(PredictedSales(X, plotSKU, 17 + w, 18 + w, strCnt))
    

historic_dates = unencoded_sales_data[( unencoded_sales_data['SKU'] == plotSKU)]['WK'].tail(weeksBkwd)
dates = historic_dates.tolist()

for i in range(weeksFwd):
    dates.append(dates[-1]+dt.timedelta(days=7))

dates
sData = historic_sales.tolist()

for r in range(len(predSales)):
    sData.append(predSales[r])

plt.style.use('_mpl-gallery')

x = dates
y = sData

fig, ax = plt.subplots(figsize=(20,9))
ax.plot_date(x, y, 'b')
ax.axvline(x=dt.date(2025, 4, 20), color = 'k')
plt.show()


# OR Model
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

def PredictedSales(df, sku, week_start, week_end, storeCount):
    total_sales = 0
    year = 2025
    input_data = df.loc[( df['SKU_'+str(sku)] == 1)].iloc[-1]
    input_data['Price %'] = .95
    input_data['Store Count'] = storeCount
    input_data['Out of Stock %'] = .10
    input_data['Out of Stock Count'] = int(storeCount*.10)
    for w in range(week_end - week_start):
        if week_start + w > 52:
            week_start -= 52
            year += 1
        input_data['week'] = week_start + w
        input_data['month'] = dt.date.fromisocalendar(year, week_start + w, 7).month
        input_data['quarter'] = pd.Timestamp(dt.date.fromisocalendar(year, week_start + w, 7)).quarter      
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = regressor.predict(input_data_reshaped)
        total_sales = total_sales + prediction
    return int(total_sales[0])

# SKU Information

sku_info = pd.read_csv('C:/Users/a/Documents/SKU_Info.csv')

skus = []
dims = []
retails = []
pCat = []
PCM = []
values = []

for s in range(len(sku_info)-1):
    if sku_info.iloc[s]['SKU'] in skus1:
        if X.loc[( X['SKU_'+str(sku_info.iloc[s]['SKU'])] == 1)].empty:
            continue
        skus.append(int(sku_info.iloc[s]['SKU']))
        dims.append([float(sku_info.iloc[s]['HGHT']), float(sku_info.iloc[s]['LGNTH']), float(sku_info.iloc[s]['WDTH'])])
        retails.append(float(sku_info.iloc[s]['Reg Retail']))
        pCat.append(str(sku_info.iloc[s]['Primary Category']))
        PCM.append(str(sku_info.iloc[s]['PCM Product Definition']))
        
    else:
        continue

skuDims = dict(zip(skus, dims))
skuRetails = dict(zip(skus, retails))
skuCats = dict(zip(skus, pCat))
skuPCM = dict(zip(skus, PCM))

perf_codes = pd.read_csv('C:/Users/a/Documents/perf_codes.csv')

perfs = []
sk = []
for s in range(len(perf_codes)-1):
    sk.append(int(perf_codes.iloc[s]['SKU']))
    perfs.append(str(perf_codes.iloc[s]['PERF']))  

skuPerfs = dict(zip(sk, perfs))
bad_perfs = ['X', 'W', 'Y', 'G']

for sku in skus:
    sales = []
    if X.loc[( X['SKU_'+str(sku)] == 1)].empty:
        skus.remove(sku)
        continue


# Iterate for different time horizons
months = ['1 month', '3 months', '6 months', '9 months']
wks = [4, 12, 16, 20]
rslt_list = []
X_list = []


for w in wks:
    values = []
    for sku in skus:
        if skuPerfs[sku] in bad_perfs:
            prediction = 0
        else:
            prediction = PredictedSales(X, sku, 17, 17+w, 887)
        values.append(prediction)
        
    skuSales = dict(zip(skus, values))
    
    # Pyomo Model
    
    model = pyo.ConcreteModel()
    
    # sets
    model.skus = pyo.Set(initialize=skuDims.keys())
    model.shelves = pyo.Set(initialize=shelfLengths.keys())
    
    # parameters
    model.width = pyo.Param(model.skus, initialize={k: skuDims[k][2] for k in skuDims})
    model.height = pyo.Param(model.skus, initialize={k: skuDims[k][0] for k in skuDims})
    model.retail = pyo.Param(model.skus, initialize={k: skuRetails[k] for k in skuRetails})
    model.sales = pyo.Param(model.skus, initialize={k: skuSales[k] for k in skuSales})
    model.shelfSpace = pyo.Param(model.shelves, initialize={k: shelfLengths[k] for k in shelfLengths})
    model.rowNum = pyo.Param(model.shelves, initialize={k: shelfLengths[k] for k in shelfLengths})
    
    # decision variables
    model.X1 = pyo.Var(model.skus, domain = pyo.Binary)
    
    # Objective Function
    def Objective_Function(model):
        return sum((model.X1[s]*model.sales[s])*model.retail[s] for s in model.skus)
    
    model.Objective = pyo.Objective(rule=Objective_Function, sense=pyo.maximize)
    
    # Shelf Space
    def shelf_space(model, r):
        return sum((model.X1[s])*model.width[s] for s in model.skus) <= 288
    
    model.c1 = pyo.Constraint(model.shelves, rule=shelf_space)
    
    # Productivity
    def productive_sku(model, s):
        return 10*(model.sales[s]*model.X1[s])/4 >= 887

    model.c2 = pyo.Constraint(model.skus, rule=productive_sku)
    
    # Solver
    solvername='glpk'
    solverpath_folder='C:/Users/a/sklearn-env/glpk-4.65/w64'
    solverpath_exe='C:/Users/a/sklearn-env/glpk-4.65/w64/glpsol' 
    solver = SolverFactory(solvername, executable=solverpath_exe)
    results = solver.solve(model)
    rslt_list.append(results)
    X_list.append(model.X1)

# lists of SKU selections
list0 = [sku for sku in X_list[0].index_set() if pyo.value(X_list[0][sku])]
list1 = [sku for sku in X_list[1].index_set() if pyo.value(X_list[1][sku])]
list2 = [sku for sku in X_list[2].index_set() if pyo.value(X_list[2][sku])]
list3 = [sku for sku in X_list[3].index_set() if pyo.value(X_list[3][sku])]

