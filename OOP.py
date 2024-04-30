import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import  preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings
import pickle as pkl

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def create_input_output(self, target_column): 
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)
        
class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModelRF() #initialize model RF
        # self.oneHotEncoder =  OneHotEncoder(sparse_output=False).set_output(transform='pandas')
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5 # Initialize 5 var split data

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)
        
    # def createMedianFromColumn(self,col):
    #     return np.median(self.x_train[col])

    def createMeanFromColumn(self,col):
        return np.mean(self.x_train[col])

    def get_mean(self, col):
        # return self.x_train[col].mean()
        self.mean_value = self.x_train[col].mean()

    def fill_na_with_mean(self,col):
        self.x_train[col].fillna(self.mean_value, inplace=True)
        self.x_test[col].fillna(self.mean_value, inplace=True)
    
    # def fillingNA(self,col,value):
    #     self.x_train[col].fillna(value, inplace=True)
    #     self.x_test[col].fillna(value, inplace=True)

    def encodeBinary(self,col):
        self.train_encode={"Gender": {"Male":1,"Female" :0}}
        self.test_encode={"Gender": {"Male":1,"Female" :0}}
        self.x_train=self.x_train.replace(self.train_encode)
        self.x_test=self.x_test.replace(self.test_encode)

    def label_encoder(self, col):
        le = LabelEncoder()
        self.x_train[col] = le.fit_transform(self.x_train[col])
        self.x_test[col] = le.transform(self.x_test[col])

    # def OneHotEncoder(self, col, drop_first=False):
    #     self.x_train = pd.get_dummies(self.x_train, columns=[col], drop_first=drop_first)
    #     self.x_test = pd.get_dummies(self.x_test, columns=[col], drop_first=drop_first)
    #     return self.x_train
    #     return self.x_test
        
    # def changeOutlierToMedian(self,kolom):
    #     Q1 = self.x_train[kolom].quantile(0.25)
    #     Q3 = self.x_train[kolom].quantile(0.75)
    #     IQR = Q3 - Q1
    #     lower_bound = Q1 - (1.5 * IQR)
    #     upper_bound = Q3 + (1.5 * IQR)
    #     self.x_train.loc[(self.x_train[kolom] < lower_bound) | (self.x_train[kolom] > upper_bound), kolom] = self.createMedianFromColumn(kolom)
        
    def removeColumn(self,col):
        self.x_train.drop(col, axis=1, inplace=True)
        self.x_test.drop(col, axis=1, inplace=True)
        
    def createModelRF(self,criteria='gini',maxdepth=4):
        self.model = RandomForestClassifier(criterion=criteria,max_depth=maxdepth)
        
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['0','1']))
        
    def train_model(self):
        self.model.fit(self.x_train, self.y_train)
    
    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 
        
    def tuningParameter(self):
        parameters = {
            'criterion':['gini', 'entropy', 'log_loss'],
            'max_depth':[2,3,4], 
        }
        RFClass = RandomForestClassifier()
        RFClass= GridSearchCV(RFClass ,
                            param_grid = parameters,   
                            scoring='accuracy',        
                            cv=5)
        RFClass.fit(self.x_train,self.y_train)
        print("Tuned Hyperparameters :", RFClass.best_params_)
        print("Accuracy :",RFClass.best_score_)
        self.createModelRF(criteria =RFClass.best_params_['criterion'],maxdepth=RFClass.best_params_['max_depth'])
        
    def createModelXGB(self):
        self.model = XGBClassifier(n_estimators=100, learning_rate = 0.1)
        
    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:
            pkl.dump(self.model, file)
            

file_path = 'data_C.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('churn')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()

#Delete variabel yang tidak digunakan
model_handler.removeColumn('CustomerId')
# model_handler.removeColumn('id')
model_handler.removeColumn('Surname')
model_handler.removeColumn('Unnamed: 0')

#Convert categorical data to numeric
model_handler.encodeBinary(['Gender'])
model_handler.label_encoder('Geography')
# model_handler.OneHotEncoder('Geography', drop_first=False)

#fill outlier with median
# model_handler.changeOutlierToMedian('Age')

#fill NA with mean
credit_score_replace_NA = model_handler.createMeanFromColumn('CreditScore')
model_handler.get_mean('CreditScore')
model_handler.fill_na_with_mean('CreditScore')
# model_handler.fillingNA('CreditScore',credit_score_replace_NA)

# #Model RF
# print("Model RF Before Tuning Parameter")
# model_handler.createModelRF()
# model_handler.train_model()
# model_handler.makePrediction()
# model_handler.createReport()

print("Model RF After Tuning Parameter")
model_handler.tuningParameter()
model_handler.train_model()
model_handler.makePrediction()
model_handler.createReport()

# #model XGB
# print("Model XGB")
# model_handler.createModelXGB()
# model_handler.train_model()
# model_handler.makePrediction()
# model_handler.createReport()

#save model
model_handler.save_model_to_file('RF_class_best.pkl')
