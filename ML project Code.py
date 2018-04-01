import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
from sklearn.model_selection import validation_curve
from sklearn.model_selection import StratifiedKFold

#set to working directory
os.chdir('C:\\Users\\Praneeth Tamvada\\PycharmProjects\\untitled3')

dinner = pd.read_csv("dinner_track.csv")
party = pd.read_csv("party_track.csv")
sleep = pd.read_csv("sleep_track.csv")
workout = pd.read_csv("workout_track.csv")


del party['uri']
del sleep['uri']
del workout['uri']
del dinner['uri']

dinner['genre'] = 1
party['genre'] = 2
sleep['genre'] = 3
workout['genre'] = 4

#Joining tables into one dataframe
frames = [dinner, party, sleep, workout]
finalaudio = pd.concat(frames)
finalaudio = finalaudio.dropna(axis=0)

audio_features = finalaudio[['acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence']]
audio_genere = finalaudio['genre']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(audio_features, audio_genere, test_size=0.3, random_state=1,stratify=audio_genere)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


## PCS D-reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2) #Create a PCA object that reduces the dimentions to 2
X_r = pca.fit(X_train).transform(X_train) #fit the data
Y = np.array(y_train)
colors = ['navy', 'turquoise', 'darkorange', 'red']
lw = 2
target_names = ['Dinner','Party','Sleep','Workout']
for color, i, target_name in zip(colors, [1, 2, 3, 4], target_names):
    plt.scatter(X_r[Y == i, 0], X_r[Y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel("Epoch")
plt.ylabel("MCC")
plt.title('PCA of Spotify feature dataset')
plt.show()



from sknn.mlp import Classifier, Layer

valid_errors = []
train_errors = []
def store_stats(avg_valid_error, avg_train_error, **_):
    valid_errors.append(avg_valid_error)
    train_errors.append(avg_train_error)

from sklearn.model_selection  import GridSearchCV


nn = Classifier(
layers=[
    Layer('Sigmoid',dropout=0.20),
    Layer("Softmax")],
    valid_size=0.2,
    callback={'on_epoch_finish': store_stats})

gs = GridSearchCV(nn, param_grid={
    'n_iter': [100,500,1000],
    'learning_rate': [0.01, 0.001],
    'hidden0__units': [10, 20, 5],
    'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]},refit=True)

gs.fit(X_train,y_train)
print(gs.best_estimator_)

plt.figure()
plt.plot(range(len(train_errors)),train_errors,color="b",label="training scores")
plt.plot(range(len(valid_errors)),valid_errors,color="r",label="validation scores")
plt.legend(loc="best")
plt.title("Train Validation Error")
plt.show()

nn_best =  Classifier(
layers=[
    Layer('Sigmoid',units=5,dropout=0.20),
    Layer("Softmax")],
    valid_size=0.2,
    n_iter=350,
    learning_rate=0.001,
    callback={'on_epoch_finish': store_stats})
nn_best.fit(X_train,y_train)


# SVM Implementation
from sklearn import svm
from sklearn.model_selection import GridSearchCV
parameter_candidates = [{'C': [1, 5, 10, 15, 20, 50, 100], 'gamma': [0.5, 0.05, 0.005, 0.01, 0.001, 0.0001, 0.00001], 'kernel': ['rbf','poly','sigmoid']}]

# Create a classifier object with the classifier and parameter candidates
svm_grid_model = GridSearchCV(estimator=svm.SVC(), param_grid= parameter_candidates, refit= True)

# Train the classifier on audio feature and target= audio genre
svm_grid_model.fit(X_train, y_train)

# View the accuracy score
print('Best score for Audio Classification:', svm_grid_model.best_score_)

# View the best parameters for the model found using grid search
print('Best C:',svm_grid_model.best_estimator_.C)
print('Best Kernel:',svm_grid_model.best_estimator_.kernel)
print('Best Gamma:',svm_grid_model.best_estimator_.gamma)

expected = y_test
predicted_nn = nn_best.predict(X_test)
predicted_svm = svm_grid_model.predict(X_test)

#plotting Confusion Matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

from sklearn.metrics import confusion_matrix

cm_nn = confusion_matrix(expected, predicted_nn)
cm_svm = confusion_matrix(expected, predicted_svm)
plot_confusion_matrix(cm_nn, classes= ['Dinner','Party','Sleep','Workout'], title='Confusion matrix for Audio Classification using Neural Network')
plot_confusion_matrix(cm_svm, classes= ['Dinner','Party','Sleep','Workout'], title='Confusion matrix for Audio Classification using Support Vector Machine')

#Zero One Loss
from sklearn.metrics import zero_one_loss
missclassification_nn = zero_one_loss(expected, predicted_nn, normalize=False)
missclassification_svm = zero_one_loss(expected, predicted_svm, normalize=False)
print("\nmissclassifications using Neural Network are",+missclassification_nn)
print("\nmissclassifications using Support Vector Machine are",+missclassification_svm)

#Classification Report
from sklearn.metrics import classification_report
classes= ['Dinner','Party','Sleep','Workout']
print("Classification report of Neural Network \n")
print(classification_report(expected, predicted_nn, target_names=classes))

print("Classification report of Support Vector Machine \n")
print(classification_report(expected, predicted_svm, target_names=classes))

#accuracy scores
from sklearn.metrics import accuracy_score
print("Accuracy of Neural Network",+accuracy_score(expected, predicted_nn))
print("Accuracy of Support Vector Machine",+accuracy_score(expected, predicted_svm))





