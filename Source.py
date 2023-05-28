import numpy as np # for linear algebra
import pandas as pd # for data processing, for data reading etc
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for plotting
#import missingno as msno # for outliers, plots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,roc_auc_score,roc_curve

path = "C:\\TTJ\\TTJ\\_TTJ_Project\\TTJ project\\dataset\\dataset.csv"
data = pd.read_csv(path)
#data.info()
#This show the information about the data

#We Check if variable has missing values
data['Attrition_Flag'].isnull().sum()
data["Attrition_Flag"].value_counts()
#We Change labels of variable "Attrition_Flag" to numeric
data["Attrition_Flag"] = data["Attrition_Flag"].map({"Existing Customer":0,"Attrited Customer":1})
data["Attrition_Flag"].value_counts()
#We fill in the missing data for categorical columns
data["Income_Category"] = data["Income_Category"].fillna("Unknown")
data["Marital_Status"] = data["Marital_Status"].fillna("Unknown")
data["Education_Level"] = data["Education_Level"].fillna("Unknown")

#We replace "abc" values in Income Category with "Unknown"
data["Income_Category"].value_counts()/len(data)*100
data["Income_Category"] = data["Income_Category"].replace('abc', 'Unknown')
data["Income_Category"].value_counts()
data.isnull().sum()

#We select the categorical variables
#Select categorical variables
categorical_columns = [col for col in data.columns if data[col].dtypes == "object"]
#print("There are", len(categorical_columns),"categorical variables. These are", categorical_columns)

# We make some graphs for the categorical columns in relationship with "Attrition_Flag"
#for col in categorical_columns:
    #fig,ax = plt.subplots(figsize = (8,6))
    #ax = sns.countplot(data = data, x = col, palette = "Set2",hue = "Attrition_Flag")
    #plt.show()

#We Check for Variables with only 1 value, and remove them.
data[categorical_columns].nunique()

#We encode for Rare Labels
values = data["Card_Category"].value_counts()/len(data)*100
data["Card_Category"] = np.where(data["Card_Category"].isin(values.index[2:]),"Superior Cards", data["Card_Category"])
data["Card_Category"].value_counts()/len(data)*100

#Numerical Variables
numerical_columns = [col for col in data.columns if data[col].dtypes != "object" and col!= "Attrition_Flag"]
#print("There are", len(numerical_columns),"numerical columns. These are", numerical_columns)

#Missing Value Imputation
#We replace the missing values for the numerical variables with the median value
data["Customer_Age"] = data["Customer_Age"].fillna(data["Customer_Age"].median())
data["Dependent_count"] = data["Dependent_count"].fillna(data["Dependent_count"].median())
data[numerical_columns].isnull().sum()

#graphics
#for col in numerical_columns:
#   fig,ax = plt.subplots(figsize = (15,6))
#   sns.histplot(data = data, x = col,hue = "Attrition_Flag",bins = 100)
#   plt.show()

#Outliers Detection
#We check for each column to see if it has outliers
q1 = data["Total_Trans_Amt"].quantile(0.25)
q3 = data["Total_Trans_Amt"].quantile(0.75)
IQR = q3 - q1
lower_limit = q1 - 3 * IQR
upper_limit = q3 + 3 * IQR

#for col in numerical_columns:
#    fig,ax = plt.subplots(figsize = (8,6))
#    sns.boxplot(y = data[col], color = 'red', whis = 3)
#    plt.show()

#We make a list with all the outliers
heavy_affected_by_outliers = ["Total_Amt_Chng_Q4_Q1","Total_Trans_Amt","Total_Ct_Chng_Q4_Q1"]

def censoring_outliers(dataframe, column):
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    IQR = q3 - q1
    lower_limit = q1 - 3 * IQR
    upper_limit = q3 + 3 * IQR
    dataframe[column] = np.where(dataframe[column] < lower_limit, lower_limit, np.where(dataframe[column] > upper_limit,upper_limit,dataframe[column]))
for variable in heavy_affected_by_outliers:
	censoring_outliers(data, variable)

#We use a table to visualize the correlation
correlation = data[numerical_columns].corr()
correlation

#We create a Heat Map
fig,ax = plt.subplots(figsize =(24,10))
sns.heatmap(correlation, annot = True, square = True,fmt = ".2f")
plt.show

#Model Development
data.to_csv("C:\\TTJ\\TTJ\\_TTJ_Project\\TTJ project\\dataset\\transformeddataset.csv") #path to dataset
data = pd.read_csv("C:\\TTJ\\TTJ\\_TTJ_Project\\TTJ project\\dataset\\transformeddataset.csv",index_col ="Unnamed: 0")

#We declare the independence variable and the target variable
#independent variables
X = data.drop(columns = ["Attrition_Flag"])

#target variable
y = data["Attrition_Flag"]
X.head()

X = X.drop(columns = ["CLIENTNUM","Total_Used_Bal"])

#We get dummies variables
X = pd.get_dummies(X, columns = categorical_columns)

#We will train and test them
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2,random_state = 10)
X_train.shape, X_test.shape,y_train.shape,y_test.shape

#We Train the Algorithm - Random Forest
#We instantiate the model
rf = RandomForestClassifier(n_estimators = 300,max_depth = 5, n_jobs = -1)
#We train the algorithm
rf.fit(X_train,y_train)

y_predict = rf.predict(X_test)
print(y_predict)

accuracy = accuracy_score(y_test,y_predict)
print("Algorithm - Random Forest")
print("Accuracy Random Forest = ", accuracy)

#We make the confusion Matrix
cm = confusion_matrix(y_test,y_predict)
fig, ax = plt.subplots(figsize = (12,8))
sns.heatmap(cm,annot = True, fmt = "d")
#plt.show()

#We get the precision and recall
precision = precision_score(y_test,y_predict)
recall = recall_score(y_test,y_predict)
print("Precision =",precision)
print("Recall = ",recall)

#We get the AUC score
auc_score = roc_auc_score(y_test,y_predict)
print("AUC Score = ",auc_score)

#We get the Roc Curve
fpr,tpr,threshold = roc_curve(y_test,y_predict)
#plt.plot(fpr,tpr)

#We Train the Algorithm - XGBoost
xgb = XGBClassifier(n_estimators = 300, max_depth = 5,learning_rate = 0.1,n_jobs =-1)
xgb.fit(X_train,y_train)

#We predict the results
y_predict = xgb.predict(X_test)
#We get the accuracy
accuracy = accuracy_score(y_test,y_predict)
print("Algorithm - XGBoost")
print("Accuracy XGBoost = ",accuracy)

#We make the confusion Matrix
cm = confusion_matrix(y_test,y_predict)
fig,ax = plt.subplots(figsize = (12,8))
sns.heatmap(cm,annot = True, fmt = "d")
#plt.show()

#We get the precision and recall
precision = precision_score(y_test,y_predict)
recall = recall_score(y_test,y_predict)
print("Precision =",precision)
print("Recall =",recall)

#We get the AUC score
auc_score = roc_auc_score(y_test,y_predict)
print("AUC Score =",auc_score)

#We get the Roc Curve
fpr,tpr,threshold = roc_curve(y_test,y_predict)
#plt.plot(fpr,tpr)

#We check for Overfitting / Underfitting
y_predict_train = xgb.predict(X_train)
y_predict_test = xgb.predict(X_test)
auc_score_train = roc_auc_score(y_train,y_predict_train)
auc_score_test = roc_auc_score(y_test,y_predict_test)
print("AUC train =",auc_score_train)
print("AUC test =",auc_score_test)

#Hyperparameters tuning
n_estimators = [200,300,400,500]
max_depth = [3,5,7 ]
learning_rate = [0.03, 0.05, 0.01]

#We search for best hyperparameters
results = []
for est in n_estimators:
    for md in max_depth:
        for lr in learning_rate:
            xgb = XGBClassifier(n_estimators = est,max_depth = md,learning_rate = lr,n_jobs =-1)
            xgb.fit(X_train,y_train)
            y_predict = xgb.predict(X_test)
            auc_score = roc_auc_score(y_test,y_predict)
            results.append(["n_estimators",est,"max_depth",md,"learning_rate",lr,"AUC",auc_score])

print(results)

best_xgb = XGBClassifier(n_estimators = 500,max_depth = 5,learning_rate = 0.05,n_jobs=-1)
best_xgb.fit(X_train,y_train)
y_predict_test = best_xgb.predict(X_test)
y_predict_train = best_xgb.predict(X_train)
auc_score_test = roc_auc_score(y_test,y_predict_test)
auc_score_train = roc_auc_score(y_train,y_predict_train)
print("Auc train =",auc_score_train)
print("Auc test =",auc_score_test)

import pickle
with open("C:\\TTJ\\TTJ\\Project_PyCharm\\best_model_xgb.pkl","wb") as file:
    pickle.dump(best_xgb,file)

y_predict = best_xgb.predict(X_test)
#print(y_predict)

#
lift_gain_report = pd.DataFrame()
lift_gain_report["y_test"] = y_test
lift_gain_report["Predicted Probabilities"] = y_predict_proba_class_1
lift_gain_report["Probabilities Rank"] = lift_gain_report["Predicted Probabilities"].rank(method ="first",ascending = True, pct = True)
lift_gain_report["Decile group"] = np.floor((1 - lift_gain_report["Probabilities Rank"]) * 10) + 1
lift_gain_report["Number of observations"] = 1
lift_gain_report = lift_gain_report.groupby(["Decile group"]).sum().reset_index()
lift_gain_report["Cumulative no. of observations"] = lift_gain_report["Number of observations"].cumsum()
lift_gain_report["Cumulative % no of observation"] = lift_gain_report["Cumulative no. of observations"]/lift_gain_report["Cumulative no. of observations"].max()
lift_gain_report["Cumulative no. of positives"] = lift_gain_report["y_test"].cumsum()
lift_gain_report["Gain"] = lift_gain_report["Cumulative no. of positives"]/lift_gain_report["Cumulative no. of positives"].max()
lift_gain_report["Lift"] = lift_gain_report["Gain"] / lift_gain_report["Cumulative % no of observation"]
print(lift_gain_report)

#We create a Lift Chart
fix, ax = plt.subplots(figsize = (15,8))
barplot = plt.bar(lift_gain_report["Decile group"], lift_gain_report['Lift'], color = 'purple')
plt.title("Lift bar plot")
plt.xlabel("Decile group")
plt.ylabel("Lift")
plt.xticks(lift_gain_report['Decile group'])

for b in barplot:
    plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,round(b.get_height(),2), ha='center')

#plt.show()


#We create a Gain Chart
lift_gain_report["Random Selection"] = lift_gain_report["Decile group"] / lift_gain_report["Decile group"].max()
fig,ax = plt.subplots(figsize=(12,8))
sns.lineplot(data = lift_gain_report, x = lift_gain_report["Decile group"],y=lift_gain_report["Gain"])
sns.lineplot(data=lift_gain_report,x =lift_gain_report["Decile group"],y=lift_gain_report["Random Selection"])
plt.title("Gain Plot")
plt.xticks(lift_gain_report["Decile group"])
plt.yticks(round(lift_gain_report["Gain"],2))
#plt.show()

#Feature Importance Analysis
feat_imp = best_xgb.get_booster().get_score(importance_type = "total_gain")
print(feat_imp)

feature_importance = pd.DataFrame()
feature_importance["Variable"] =feat_imp.keys()
feature_importance["Importance value"]=feat_imp.values()
feature_importance["Importance value %"] = feature_importance["Importance value"] / feature_importance["Importance value"].sum() * 100
feature_importance.sort_values(by = ["Importance value"],ascending = False)

#Shap Chart
import shap
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values,X_test,plot_size = (20,10))