import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')
print(df.describe().T)

# check for missing data
print(df.isnull().sum())
print(df.dtypes)
sns.countplot(x='diagnosis', data=df)

# replace categorical values with numbers
print('Distribution of data: ', df['diagnosis'].value_counts())

# define the dependent variable that needs to be predicted (diagnosis)
y = df['diagnosis'].values
print(y)
print('Labels before encoding are: ', np.unique(y))

# encoding categorical data from text (B, M) to integers (0, 1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(y)
print('Labels after encoding are: ', np.unique(Y))

# define and normalize/scale values
X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
print(X.describe().T) #needs scaling

# X needs scalling, bring everything in the same range
# scale/normalize the values to bring them to similar range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X) 

# split data into training and testing to verify accuracy after fitting the model
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=42)

# check the shape of the traing and testing data
print('Shape of training data is: ', X_train.shape) 
print('Shape of testing data is: ', X_test.shape)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

# defining the model
model = Sequential()
model.add(Dense(16, input_dim=30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# fit the model with no-early stopping or other callbacks
# history is a dictionary to capture the loss and accuracy scores so i can plot later on
history = model.fit(X_train, Y_train, verbose=1, epochs=100, batch_size=64, validation_data=(X_test, Y_test))

# plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Taining Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Taining Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predicting the test set results
y_pred = model.predict(X_test)
print(y_pred)
y_pred = (y_pred > 0.5)
print(y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
sns.heatmap(cm, annot=True)
