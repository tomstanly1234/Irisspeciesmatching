import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling.visualisation.plot import correlation_matrix

iris=pd.read_csv('Iris.csv',index_col=0)
species=iris['Species']
le=LabelEncoder()
speciesbynum=le.fit_transform(species)
parameters=iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
X_train,X_test,Y_train,Y_test=train_test_split(parameters,speciesbynum,test_size=0.2,random_state=42)
dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)
predictionofdecisiontree=dt.predict(X_test)
rf=RandomForestClassifier()
rf.fit(X_train,Y_train)
predictionofrf=rf.predict(X_test)
supportvector=SVC()
supportvector.fit(X_train,Y_train)
predictionofsvm=supportvector.predict(X_test)
print('accuracy of dt',accuracy_score(predictionofdecisiontree,Y_test))
print('accuracy of svm',accuracy_score(predictionofsvm,Y_test))
print('accuracy of rf',accuracy_score(predictionofrf,Y_test))
plt.scatter(x=iris['Species'],y=iris['SepalLengthCm'])
plt.show()
plt.scatter(x=iris['Species'],y=iris['PetalLengthCm'])
plt.show()
correlation_matrix=iris.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
