
## Table of Contents

1. Motivation
2. File Descriptions
3. How to interact with the Project
4. Licensing and Acknowledgements

### 1. Motivation
The objective of the project is to build a NLP based classification of the disaster recovery messages. There are total of 36 categories of messages. The training is based on the provided files containing messages and categories. The project also includes the app where a user can enter a message and get the categories for this message.

### 2. File Descriptions

app - directory contains web app code and templates
app/templates
app/templates/go.html
app/templates/master.html
app/run.py - code to start and run web application

data - directory contains data and MLE trainig pipeline
data/categories.csv - trainig and testing data for categories
data/messages.csv - training and testing data for messages
data/process_data.py - ETL pipeline that cleans data and stores in a sqlight database

models - directory contains training code
models/train_classifier.py - training code that loads the data pre-processed in the previous step and produces classification model

### 3. How to Interact with the Project

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

### 4. Licensing and Acknowledgements

Code was created based on the examples fron Udacity Data Scientist nano degree course. 
https://plotly.com/ was used to learn how to sest up charts for the web application

