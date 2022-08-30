#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 20:06:36 2021

@author: a
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

import plotly.express as px
import plotly.graph_objects as go



st.title("Customer Personality Analysis")

# giving the webpage a title

#st.header("This application helps you identify Customer Personality Analysis which is a detailed analysis of a company’s ideal customers")
st.header("Model Deployment")
# here we define some of the front end elements of the web page like 
# the font and background color, the padding and the text to be displayed
html_temp = """
<div style ="background-color:grey;padding:3px">
<h1 style ="color:white;text-align:center;">Group - 6  Batch - P76 </h1>
</div>
"""
# this line allows us to display the front end aspects we have 
# defined in the above code
st.markdown(html_temp, unsafe_allow_html = True)
result =""


st.sidebar.header('About ')
st.sidebar.header("This application helps you identify Customer Personality Analysis which is a detailed analysis of a company’s ideal customers")



data = pd.read_csv(r'/Users/a/Desktop/marketing_campaign.csv')

data['NumAllPurchases'] = data['NumDealsPurchases'] + data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']
data['Total_Expenses'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']
data['Campaign'] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data['AcceptedCmp4'] + data['AcceptedCmp5']


data['Age'] = 2021 - data['Year_Birth']
data['Age'] = data['Age'].replace(np.NaN, data['Age'].mean())
data[['Age']]=data['Age']
data = data.assign(Age = pd.cut(data['Age'], 
                               bins=[ 0 , 19 , 20, 40 ,60], 
                               labels=['Below 19', '20 - 39', '40 - 59','Above 60']))
#data = data.drop("Age", axis = 1)

data['Income'] = data['Income'].replace(np.NaN, data['Income'].mean())
data = data.assign(Incomes=pd.cut(data['Income'], 
                               bins=[ 0, 25000, 50000,100000,666666], 
                               labels=['Below 25000', 'Income 25000-50000 ', 'Income 50000-100000 ','Above 100000']))
data = data.drop("Income", axis = 1)

data['Total_Expenses'] = data['Total_Expenses'].replace(np.NaN, data['Total_Expenses'].mean())
data = data.assign(Total_Expenses = pd.cut(data['Total_Expenses'], 
                               bins=[ 0, 500, 1000, 2000, 2525], 
                               labels=['Below 500', 'Expense 500-1000 ','Expense 1000-2000','Above 2000']))

data['Marital_Status'] = data['Marital_Status'].replace(['Married', 'Together'], 'Relationship')
data['Marital_Status'] = data['Marital_Status'].replace(['Single', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO'], 'Single')




data['Education'].replace({'2n Cycle' : 'Master'},inplace = True)
data['Education'] = data['Education'].replace(['Graduation', 'Master'], 'Graduated')
data['Education'] = data['Education'].replace(['PhD'], 'PHD')

data['Children'] =data.Kidhome + data.Teenhome

data = data.drop(columns = ["ID","Year_Birth","Dt_Customer","Recency",'Kidhome',
       'Teenhome',"MntWines","MntFruits","MntMeatProducts","MntFishProducts","MntSweetProducts","MntGoldProds","AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5","NumStorePurchases","NumDealsPurchases","NumWebPurchases","NumCatalogPurchases",'NumWebVisitsMonth',"Z_CostContact","Z_Revenue"],axis = 1)




for feature in ['Education','Marital_Status','Age','Incomes','Total_Expenses']:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])


from sklearn.preprocessing import normalize
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns = data.columns)


from sklearn.cluster import AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters = 4,affinity = 'euclidean',linkage = "ward") 
model_hc = agg.fit_predict(data_scaled)
clusters = pd.DataFrame(model_hc,columns =['Clusters'])
data['Cluster_id'] = agg.labels_
data.groupby("Cluster_id").agg(['mean']).reset_index()

X = data.drop("Cluster_id", axis=1)
y = data.Cluster_id
X.shape, y.shape


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size = 0.2, random_state = 10)
from sklearn.ensemble import RandomForestClassifier 
model = RandomForestClassifier(max_depth=4, random_state = 10) 
model.fit(x_train, y_train)


from sklearn.metrics import accuracy_score
pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)

pred_train = model.predict(x_train)
accuracy_score(y_train,pred_train)


# saving the model 
import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()


# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)



@st.cache()

# defining the function which will make the prediction using the data which the user inputs 
def prediction(Education,Marital_Status,Complain,Response,NumAllPurchases,Total_Expenses,Campaign,Age,Incomes,Children):   
 
    # Pre-processing user input  
    
    if Age ==   "Below 19":
         Age = 0
    elif Age == '20 - 39':
        Age = 1
    elif Age == '40 - 59':
        Age = 2  
    elif Age == 'Above 60':
        Age = 3
    
    if Education == "Basic":
        Education = 0
        
    elif Education == "Master":
        Education = 1
        
    elif Education == "Graduated":
        Education = 2
        
    elif Education == "PHD":
        Education = 3

        
    if Marital_Status == "Single":
        Marital_Status = 0
    
    elif Marital_Status == "Relationship":
        Marital_Status = 1
        
   
        
    if  Children == "0":
        Children = 0
    elif Children == "1":
        Children = 1
    elif Children == "2":
        Children = 2
    elif Children == "Above 2":
       Children = 3

    if Incomes == "Below 25000":
        Incomes = 0
    
    elif Incomes == "Income 25000-50000":
        Incomes = 1
        
    elif Incomes == "Income 50000-100000":
        Incomes = 2
        
    elif Incomes == "Above 100000":
        Incomes = 3
        
    if Total_Expenses == "Below 500":
        Total_Expenses = 0
    elif Total_Expenses == "Expense 500-1000":
        Total_Expenses = 1
    elif Total_Expenses == "Expense 1000-2000":
        Total_Expenses = 2
    elif Total_Expenses == "Above 2000":
        Total_Expenses = 3
        
        
        
    if Complain == "NO":
        Complain = 0
    
    elif Complain == "YES":
        Complain = 1
        
        

    if Campaign == "AcceptedCampaign 0":
        Campaign = 0
    
    elif Campaign == "AcceptedCampaign 1":
        Campaign = 1
        
    elif Campaign == "AcceptedCampaign 2":
        Campaign = 2
        
    elif Campaign == "AcceptedCampaign 3":
        Campaign = 3  
        
    elif Campaign == "AcceptedCampaign 4":
        Campaign = 4
        


  
    if Response == "NO":
        Response = 0
    
    elif Response == "YES":
        Response = 1


    
 

    prediction = classifier.predict( 
        [[Education,Marital_Status,Complain,Response,NumAllPurchases,Total_Expenses,Campaign,Age,Incomes,Children]])
            
    if prediction == 0:
        pred = 'cluster 0'
   
    elif prediction == 1:
        pred = 'cluster 1'
    
    elif prediction == 2:
        pred = 'cluster 2'
    return pred

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
   
   
   
    # display the front end aspect
    #st.markdown(html_temp, unsafe_allow_html = True) 
    #st.sidebar.header('Input Data')
   # st.markdown(html_temp, unsafe_allow_html = True)
    #result =""

    
  
    # following lines create boxes in which user can enter data required to make prediction 
    
    Age = st.selectbox("Age", ('Below 19', '20 - 39', '40 - 59','Above 60'))
    
    Education = st.selectbox("Education",("Basic","Master","Graduated","PHD"))
    
    Marital_Status = st.radio("Marital_Status: ", ('Single', 'Relationship'))
    if (Marital_Status == 'Single'):
        st.success("Single")
    elif (Marital_Status == 'Relationship'):
        st.success("Relationship")
    
    
    
   
    
    Children = st.selectbox("No of Childrens : ",( "0",'1','2','Above 2'))
    
    Incomes = st.selectbox("Incomes",("Below 25000", "Income 25000-50000", "Income 50000-100000","Above 100000")) 
   
    Total_Expenses = st.selectbox("Total_Expenses", ("Below 500", 'Expense 500-1000 ','Expense 1000-20000','Above 2000'))
    
    Num_of_All_Purchases = st.slider("Number of Purchases :", 0, 50)
    st.text('Selected: {}'.format(Num_of_All_Purchases)) 
    
    #Total_Expenses = st.slider("Total_Expenses :", 0, 3000)
    #st.text('Selected: {}'.format(Total_Expenses)) 
    
    
    Complain = st.selectbox("If Customer Complained in the last 2 years",("YES","NO"))
    
    Campaign = st.selectbox("Campaign",("AcceptedCampaign 0","AcceptedCampaign 1","AcceptedCampaign 2","AcceptedCampaign 3","AcceptedCampaign 4"))
    
    
    
    Response = st.selectbox("If Customer accepted the offer in the last Campaign",("YES","NO"))
    
    result =""
          
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Education,Marital_Status,Complain,Response,Num_of_All_Purchases,Total_Expenses,Campaign,Age,Incomes,Children) 
        st.success('Common cluster is {}'.format(result))
   
     
if __name__=='__main__': 
    main()





