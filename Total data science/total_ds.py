import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from PIL import Image

#setting title
st.title('Total data science')
image=Image.open('tdslogo.png')
st.image(image,use_column_width=True)

def main():
	activities=['EDA','Visualization','Model','About us']
	option=st.sidebar.selectbox('Selection option:',activities)

	#Dealing with the EDA part
	if option=='EDA':
		st.subheader("Exploratory Data Analysis")
		data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
		st.success("Data successfully loaded")
		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head(10))
			if st.checkbox("Display shape"):
				st.write(df.shape)
			if st.checkbox("Display columns"):
				st.write(df.columns)
			if st.checkbox("Select multiple columns"):
				selected_columns=st.multiselect("Select preferred columns",df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)

			if st.checkbox("Display summary"):
				st.write(df.describe().T)

			if st.checkbox('Display Null values'):
				st.write(df.isnull().sum())

			if st.checkbox("Display Correlation of data various columnns"):
				st.write(df.corr())

	#Dealing with the visualization part
	elif option=='Visualization':
		st.subheader('Data Visualization')
		data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
		st.success("Data successfully loaded")
		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head(10))	

			if st.checkbox('Select multiple columns to plot'):
				selected_columns=st.multiselect('Select your preferred columns',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)

			if st.checkbox('Display Heatmap'):
				st.write(sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
				st.pyplot()

			if st.checkbox('Display Pairplot'):
				st.write(sns.pairplot(df1,diag_kind='kde'))
				st.pyplot()

			if st.checkbox('Display Piechart'):
				all_columns=df.columns.to_list()
				pie_columns=st.selectbox('select column to display',all_columns)
				pieChart=df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
				st.write(pieChart)
				st.pyplot()

			
	#Dealing with the model building part
	elif option=='Model':
		st.subheader('Data Visualization')
		data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
		st.success("Data successfully loaded")
		if data is not None:
			df=pd.read_csv(data)
			st.dataframe(df.head(10))

			if st.checkbox('Select multiple columns'):
				new_data=st.multiselect('Select your preferred columns, your last selected column will be the target column',df.columns)	
				df1=df[new_data]
				st.dataframe(df1)

				#diving the data into X and Y variables
				X=df1.iloc[:,0:-1]
				y=df1.iloc[:,-1]

			seed=st.sidebar.slider('Seed',1,200)

			classifier_name=st.sidebar.selectbox('Select your preferred classifier',('KNN','SVM','LogisticRegression','Naive Bayes','Decision Tree'))

			def add_parameter(name_of_classifier):
				params=dict()
				if name_of_classifier=='SVM':
					C=st.sidebar.slider('C',0.01,15.0)
					params['C']=C

				else:
					name_of_classifier=='KNN'
					K=st.sidebar.slider('K',1,15)
					params['K']=K
				
				return params

			#calling the function

			params=add_parameter(classifier_name)

			#defining a function for our classifier

			def get_classifier(name_of_classifier,params):
				clf=None
				if name_of_classifier=='SVM':
					clf=SVC(C=params['C'])
				elif name_of_classifier=='KNN':
					clf=KNeighborsClassifier(n_neighbors=params['K'])
				elif name_of_classifier=='LogisticRegression':
					clf=LogisticRegression()
				elif name_of_classifier=='Naive Bayes':
					clf=GaussianNB()
				elif name_of_classifier=='Decision Tree':
					clf=DecisionTreeClassifier()
				else:
					st.warning('Select your choice of algorithm')

				return clf

			clf=get_classifier(classifier_name,params)

			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=seed)

			clf.fit(X_train,y_train)
			y_pred=clf.predict(X_test)	
			st.write('Predictions:',y_pred)
			accuracy=accuracy_score(y_test,y_pred)	
			st.write('Name of Classifier:',classifier_name)
			st.write('Accuracy:',accuracy)



	#Dealing with background information
	elif option=='Background information':
		st.write('This is an interactive web page for the ML projects, feel free to use it')
		st.baloons()

		




if __name__=='__main__':
	main()


