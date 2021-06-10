#Start
import pandas as pd
pd.options.display.max_columns = 100

address = "D:/LUMSU/Zrogramming/Data Science/Kaggle Projects/Data/World Happiness Report/"
addressTest = address + "2018.csv"
addressTrain = address + "2019.csv"

# Import Titanic Testing Data
test = pd.read_csv(addressTest,index_col=0)
#Import Titanic Training Data
trainFull = pd.read_csv(addressTrain,index_col=0)

# Printing Some Basic Information
printBasicInfo = False # Print head and missing values at the start
missing_values_count = trainFull.isnull().sum()
if printBasicInfo:
    print( "Head\n" , trainFull.head() )
    print("\nOverall Shape: ", trainFull.shape, "\n" )
    print( "Missing values in each column\n", missing_values_count)
    
# Assigning X and y
y_trainFull = trainFull["Score"]
X_columns = [ "GDP per capita","Social support",'Healthy life expectancy',"Freedom to make life choices","Generosity","Perceptions of corruption"]             
X_trainFull = trainFull[X_columns]

# Spliting Data into Training and Validation
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X_trainFull,y_trainFull,random_state = 0)

from xgboost import XGBRegressor
  
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(train_X, train_y, 
             early_stopping_rounds=5, 
             eval_set=[(val_X, val_y)], 
             verbose=False)

predictionsVal = my_model.predict(val_X)
predictionsTest = my_model.predict(test[X_columns])    
    

from sklearn.metrics import mean_absolute_error
print("Validation Data (Mean Absolute Error): " + str(mean_absolute_error(predictionsVal, val_y)))
print("Test Data (Mean Absolute Error): " + str(mean_absolute_error(predictionsTest, test["Score"])))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16,9))
sns.regplot( x=val_y, y=predictionsVal )
plt.figure()

plt.figure(figsize=(16,9))
sns.regplot( x=test["Score"], y=predictionsTest ) 
plt.figure()


#import matplotlib.pyplot as plt
#import seaborn as sns

# Set the width and height of the figure
#plt.figure(figsize=(16,9))
# Add title
#plt.title("Daily Global Streams of Popular Songs in 2017-2018")

#sns.lineplot( data=trainFull[{"New Price","Price"}] )
#plt.figure()

#sns.scatterplot( x=trainFull["Milleage"], y=trainFull["Ratio"] )
#plt.figure()

#sns.scatterplot( x=trainFull["Price"], y=trainFull["New Price"], hue=trainFull["Company" ])
#plt.figure()

#sns.regplot( x=trainFull["Milleage"], y=trainFull["Ratio"] )
#plt.figure()

#sns.kdeplot( x=trainFull["Milleage"], y=trainFull["Ratio"] )
#plt.figure()

#plt.hist(trainFull["New Price"])
#plt.figure()

#plt.hist(trainFull["Price"])
#plt.figure()












