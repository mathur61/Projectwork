import pandas
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
# print(dataset_2.to_string()) #This line will print out your data


#QUESTION 1
dataset_2['Total Traffic'] = dataset_2['Brooklyn Bridge'] + dataset_2['Manhattan Bridge'] + \
                             dataset_2['Queensboro Bridge'] + dataset_2['Williamsburg Bridge'] #claculates the total traffic across all the bridges and makes a new column for it  
                              

def threeBridgeModels(bridge1, bridge2, bridge3):
    X = dataset_2[[f"{bridge1} Bridge", f"{bridge2} Bridge", f"{bridge3} Bridge"]].values #creating our input and the target variable with all the bridges being stored in X variable 
    y = dataset_2["Total Traffic"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #split it into training and testing, and using 30% of the ata for it, and other 70% will be used for training this model
    model = LinearRegression() #creating the model we want to create 
    model.fit(X_train, y_train) #train with the training data we found
    y_pred = model.predict(X_test) #the trained model is used to predict our value and stored 
    mse = mean_squared_error(y_test, y_pred) #we use our testing data and predicted data to calculate the average squared different between the values, smaller MSE better the  model 
    
    return model, mse

bridge_combinations = [("Brooklyn", "Manhattan", "Williamsburg"),
                       ("Brooklyn", "Manhattan", "Queensboro"),
                       ("Brooklyn", "Williamsburg", "Queensboro"),
                       ("Manhattan", "Williamsburg", "Queensboro")]  #a list of tuples with the possbile combinations that we can put the cameras in

#initializing all the variables to find our concluding results
best_mse = float('inf')
best_combination = None

for combination in bridge_combinations: #loop that iterates over the combinationsand obtains the linear regresion model and MSE for each
    bridge1, bridge2, bridge3 = combination
    model, mse = threeBridgeModels(bridge1, bridge2, bridge3)
    
    print(f"Combination: {bridge1}, {bridge2}, {bridge3}")
    print("Mean Squared Error:", mse)
    
    if mse < best_mse: #to find the best one we are finding the lowwest one and that is the best bridge combination
        best_mse = mse
        best_combination = combination

print("\nBest Combination of Bridges:", best_combination)
print("Best Mean Squared Error:", best_mse)


#QUESTION 2

rowSize = dataset_2.shape[0] #assinging the rowsize

dataset_2['Precipitation'] = pandas.to_numeric(dataset_2['Precipitation'].replace('T', '0.00')) #converting the values to numeric and any T values to 0 incase

for i in range(5, 10): #iterates from 5th to 10th column, to make sur all of it is numeric
    if not pandas.api.types.is_numeric_dtype(dataset_2.iloc[:, i]): #if it is not it replaces to change it to numeric specifically integer as it is traffic
        dataset_2.iloc[:, i] = dataset_2.iloc[:, i].str.replace(',', '').astype(int)

for i in range(2, 5):  #this does the same, iterates over the columns to make sure it is a float, which are the temperatures
    if not pandas.api.types.is_numeric_dtype(dataset_2.iloc[:, i]): #if it not, it converts it using integer location in the next step 
        dataset_2.iloc[:, i] = dataset_2.iloc[:, i].astype(float) #used to convert it to float as they are temperature related parameters


X = dataset_2[['High Temp', 'Low Temp', 'Precipitation']].values #extracting and making an array of these 

scaler = StandardScaler() #scales it like a normal distrubution 
X_scaled = scaler.fit_transform(X) #to fit our data to the scaler, by finding how much it needs to be transformed, and the new data


n_clusters = 3  #the algorithm will find 3 clusters in the data
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10) #set n_init to ignore a warning, rest of it is initialized to make sure the clusterin is done properlu
dataset_2['Weather Cluster'] = kmeans.fit_predict(X_scaled) #applied to the scaled data we found, fitting it to the kmeans algorithm 

print(dataset_2['Weather Cluster'])

label_encoder = LabelEncoder() #converting the category to numerical values
dataset_2['Weather Cluster'] = label_encoder.fit_transform(dataset_2['Weather Cluster']) #replacing with numerical labels now, to find each category of the diffrent temperature/weather types

X = dataset_2[['High Temp', 'Low Temp', 'Precipitation', 'Weather Cluster']] #input for the model we want, to predict the traffic throughout 
y = dataset_2['Total'] #target variable being the total traffic 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) #testing the model with 30% of the data and training it with the rest

#converting the data to arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

regression_models = {} #initalizing dictionary to store the cluster models 


for cluster in dataset_2['Weather Cluster'].unique(): #iterating over each cluster, filtering through it
    X_cluster = X_train[X_train[:, 3] == cluster]  
    y_cluster = y_train[X_train[:, 3] == cluster]
    
    regression_model = LinearRegression() #initializng regression model
    
    regression_model.fit(X_cluster[:, :3], y_cluster)  # training the model for current cluster
    
    regression_models[cluster] = regression_model #cluster is the key, and the model is stored properlu

predictions = [] #for the list of predictions we find

for i in range(len(X_test)):

    current_cluster = int(X_test[i, 3])  #finding the weather cluster for each data point
    current_features = X_test[i, :3] #corresponding temperature features that match
    prediction = regression_models[current_cluster].predict([current_features]) #making predictions based on what we found
    predictions.append(prediction[0]) #adding it to the prediction list

predictions = np.array(predictions) #converting it to an array 

r2 = r2_score(y_test.values, predictions)
print("R-squared:", r2)

#QUESTION 3

day_label_encoder = LabelEncoder() #initializing creating a new class
dataset_2['Day Label'] = day_label_encoder.fit_transform(dataset_2['Day']) #fitting it to unique days of the week from 0-6 for each day of the week and assigning numeric labels 

X = dataset_2[['Total Traffic']].values #making the traffic specific column from the array the x variable for regression
y = dataset_2['Day Label'].values #the target variable being the day of the week

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #splitting the data for testing and training, using 42 to randomize

model = MultinomialNB() #we use multinomial naive bayes to analyse this prediction of the day based on the traffic

model.fit(X_train, y_train) #training the classifier with 70% of the data

y_pred = model.predict(X_test) #assinging the predicted values for us to finally evaluate if it is feasible to make this prediction based on the data given

new_data_point = X_test[0]  #using this new data point from the testing set,
predicted_day_index = model.predict([new_data_point]) #making the prediction for this data point 
predicted_day = day_label_encoder.inverse_transform(predicted_day_index) #assinging the day this datapoint would correspond to using inverse transform 
print("The predicted day is:", predicted_day[0]) #this then just predicts the day 

r2 = r2_score(y_test, y_pred) #this gives us the r score for the y actual values and predicted values 
print("R-squared:", r2) #in order to evaluate if we could use this to analyse


#plottting the weather clusters to visualize trends and see if we can it to predict 
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dataset_2['Weather Cluster'], cmap='rainbow', edgecolor='k', s=50)
plt.xlabel('Standardized High Temp')
plt.ylabel('Standardized Low Temp')
plt.title('Weather Clusters')
plt.colorbar(label='Cluster')
plt.show()
