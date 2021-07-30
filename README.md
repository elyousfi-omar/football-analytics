# Football Analytics Application
## Introduction
This project was a result of my Master Thesis: 'Design and implementation of football data analysis application'. In my thesis I talked in detail about the history of data analytics in sports, the importance of good football analysis to get meaningful conclusions, the multiple applications of data analytics in sports.
In this readme file I will explain briefly the steps I took to create the application; The application is written in python, and the interface was created using Tkinter.
## Data
The data used in this project is a sample offered by StatsBomb for research purposes. The data repository: https://github.com/statsbomb/open-data
I have used events and match data to analyze:
* Performance of both teams in a certain match.
* Performance of a player in a certain match.
* Performance of a team in a certain season.
In order to create these functionnalities, the step of pre processing is very important.

## Pre processing
In the preprocessing 'preprocessing.ipynb' notebook file, I explained the steps I took to create:
* Shots dataset of a team in a certain season.
* Number of matches of each team in the whole dataset.
* Goalscorers of a team in a certain season.
* The teams and competition of each match id.
* Machine learning model (Expected goals model): The goal of this model is to predict the probability of a shot resulting in a goal.
* Training and tuning the model using GridSearchCV.

These weren't the only pre processing tasks but they are the most important.

## Interface
I created the simple interface using Tkinter. I uploaded a file containing the whole code: 'Interface.py'
The code is kind of simple, every function is self explanatory and accompanied with a comment.

#### Note
I have used some code created by https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython/blob/master/FCPython.py
Make sure to check them out.
## Some of the results
![Main interface](match_analysis.png?raw=true "Title")

