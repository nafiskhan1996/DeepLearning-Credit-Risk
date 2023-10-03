#!/usr/bin/env python
# coding: utf-8

# Import Necessary Libraries
# 

# In[14]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV  # Ensure this line is present and correct
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# Load the Dataset

# In[2]:


df = pd.read_csv('credit_risk_dataset.csv')


# Explore the Dataset

# In[3]:


print(df.head())


# Handle Missing Values

# In[4]:


print(df.isnull().sum())  # Check for missing values
df = df.dropna()  # Drop missing values or use imputation as appropriate


# Encode Categorical Features

# In[5]:


df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent', 'loan_grade'], drop_first=True)


# In[6]:


df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})


# In[7]:


print(df['cb_person_default_on_file'].unique())  # It should print [1 0]


# Split Data

# In[8]:


from sklearn.model_selection import train_test_split

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


# Set up parameter grid for hyperparameter tuning
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

# Create a GridSearchCV object
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

# Print best parameters
print(grid.best_params_)

# Predict on the test data
grid_predictions = grid.predict(X_test)

# Print classification report
print(classification_report(y_test, grid_predictions))


# In[16]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import numpy as np

param_dist = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6), 'kernel': ['rbf']}

# Using fewer iterations and all CPU cores
random_search = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=10, cv=5, verbose=2, n_jobs=-1)
random_search.fit(X_train, y_train)


# Normalize Data

# In[19]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[20]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Create a Sequential model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])


# Build the Model

# In[21]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the Model

# In[22]:


model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Evaluate the Model

# In[23]:


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy*100:.2f}%")


# Confusion Matrix and Classification Report

# In[24]:


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)  # convert probabilities to binary output

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
cr = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(cr)


# In[ ]:




