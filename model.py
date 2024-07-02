# Note that this file is the model code of the file labelled admission_model.pkl
# the pkl file can be generated using tools like pickle and joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

admission_predict = pd.read_csv('Dataset.csv',index_col=0)

features = ['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA']

x = admission_predict[features]

y = admission_predict['Chance of Admit ']

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=9,test_size=0.25)

model = LinearRegression()

model.fit(x_train,y_train)

pred = model.predict(x_test)

R2_score = r2_score(y_test,pred)

# calculating accuracy and error factors
print('MSE : mean squared error : ',mean_squared_error(y_test,pred))

print('RMSE : root mean squared error : ',(mean_squared_error(y_test,pred))**0.5)

print('r2 score is: ' + str(R2_score) + ' Accuracy is: ' + str(round(R2_score*100,2)))

# initializing random test scores and other ratings
GRE_Score=305
TOEFL_Score = 108 
University_Rating = 4
SOP = 4.5 
LOR = 4.5
CGPA = 8.35


print('Your chances of getting admission is : {}%'.format(round(model.predict([[GRE_Score,TOEFL_Score,University_Rating,SOP,LOR,CGPA]])[0]*100, 1)))