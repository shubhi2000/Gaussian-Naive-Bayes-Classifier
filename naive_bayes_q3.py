from sklearn.datasets import load_wine
from sklearn import naive_bayes
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import math


#loading the given dataset
X,Y=load_wine(return_X_y=True)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

#array for storing predictions by Gaussian Naive Bayes
predictions = [0]*(Y_test.shape[0])
predictions=np.zeros_like(predictions)


conf_mat=[[0,0,0],[0,0,0],[0,0,0]]
conf_mat=np.zeros_like(conf_mat)

def prob(h,x):
  """ Function to find probability that the hypothesis h 
   is true for the feature values in array x
   h=0, 1 or 2 in our case
   x is a test sample
   """
  
  # initialisation for mean for each feature for those training samples who label is h
  mean=[0]*X_train.shape[1]

  # initialisation for variance for each feature for those training samples who label is h
  var=[0]*X_train.shape[1]

  # count of features in training set having label = h 
  num=0

  for j in range(X_train.shape[0]):
    if Y_train[j]==h:
      num+=1
      for k in range(X_train.shape[1]):
        mean[k]+=X_train[j][k]
        var[k]+=(X_train[j][k]**2)
  
  # gaussian for each feature 
  gaussian=[0]*X_train.shape[1]

  for j in range(X_test.shape[1]):
    if num!=0:
      mean[j]=mean[j]/num
      var[j]=var[j]/num-mean[j]**2

    gaussian[j]=(1/((2*math.pi*var[j])**0.5))*math.exp(-1*((x[j]-mean[j])**2/(2*var[j])))

  # probability for hypothesis h to be considered true  
  probability=1

  for j in range(X_test.shape[1]):
    probability*=gaussian[j]

  probability*=num/X_train.shape[0]

  return probability

correct_=0

# calculating prediction for each test sample
for i in range(Y_test.shape[0]):
  x=X_test[i,:]

  # Case: Hypothesis = 0
  p=prob(0,x)
  c=0    # c denotes current prediction
 
  # Case: Hypothesis = 1
  p1=prob(1,x)

  if p1>p:
    p=p1
    c=1
  
  # Case: Hypothesis = 2
  p1=prob(2,x)
  if p1>p:
    p=p1
    c=2

  if c==Y_test[i]:
    correct_+=1
    
  conf_mat[Y_test[i],c]+=1


  # storing the final prediction
  predictions[i]=c

print("Actual results")
print(Y_test)

print()


print("Predicted results")
print(predictions)
print()

print("Evaluation metrics for implemented model ")
print()

print("Accuracy")
print(correct_/Y_test.shape[0])
print("Confusion matrix")
print(conf_mat)


# class = 0
F1_0=conf_mat[0,0]/(conf_mat[0,0]+0.5*(conf_mat[1,0])+conf_mat[2,0]+conf_mat[0,1]+conf_mat[0,2])
# class = 1
F1_1=conf_mat[1,1]/(conf_mat[1,1]+0.5*(conf_mat[0,1])+conf_mat[2,1]+conf_mat[1,0]+conf_mat[1,2])
# class = 2
F1_2=conf_mat[2,2]/(conf_mat[2,2]+0.5*(conf_mat[0,2])+conf_mat[1,2]+conf_mat[2,0]+conf_mat[2,1])

print("Classwise F1 score value")
print("Class 0 " +str(F1_0))
print("Class 1 "+str((F1_1)))
print("Class 2 "+str((F1_2)))
print("F1-score")
print((F1_0+F1_1+F1_2)/3)

print()
# implementation from inbuilt gaussian
model=naive_bayes.GaussianNB()
model.fit(X_train,Y_train)
print("Predicted results using inbuilt")
pred=model.predict(X_test)
print(pred)

print()
print("Evaluation metrics for sklearn inbuilt model")
print()
print("Accuracy")
print(metrics.accuracy_score(Y_test,pred))
print("Confusion Matrix")
conf_mat=metrics.confusion_matrix(Y_test,pred)
print(conf_mat)

# class = 0
F1_0=conf_mat[0,0]/(conf_mat[0,0]+0.5*(conf_mat[1,0])+conf_mat[2,0]+conf_mat[0,1]+conf_mat[0,2])
# class = 1
F1_1=conf_mat[1,1]/(conf_mat[1,1]+0.5*(conf_mat[0,1])+conf_mat[2,1]+conf_mat[1,0]+conf_mat[1,2])
# class = 2
F1_2=conf_mat[2,2]/(conf_mat[2,2]+0.5*(conf_mat[0,2])+conf_mat[1,2]+conf_mat[2,0]+conf_mat[2,1])

print("Classwise F1 score value")
print("Class 0 " +str(F1_0))
print("Class 1 "+str((F1_1)))
print("Class 2 "+str((F1_2)))
print("F1-score")
print((F1_0+F1_1+F1_2)/3)
