import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

#importing datasets
nfl2021 = pd.read_csv('NFLData.csv', index_col=0)
nfl2021['Year'] = 2021
week13data = pd.read_csv('NFLDataWeek13.csv', index_col=0)
week13data['Year'] = 2021
nfl2020 = pd.read_csv('NFLData2020.csv', index_col=0)
nfl2020['Year'] = 2020
nfl2019 = pd.read_csv('NFLData2019.csv', index_col=0)
nfl2019['Year'] = 2019
nfl2018 = pd.read_csv('NFLData2018.csv', index_col=0)
nfl2018['Year'] = 2018
games = pd.read_csv('games.csv')
experts = pd.read_csv('ExpertsPicks.csv')

dfexperts = pd.DataFrame(experts)
xexperts = list(dfexperts.iloc[:, 0])
yexperts = list(dfexperts.iloc[:, 3])
plt.figure(figsize=(12,12))
plt.bar(xexperts, yexperts, color='g')
plt.title("Accuracy of Experts Picks in the 2021 NFL Season")
plt.xlabel("Experts Name")
plt.ylabel("Accuracy of Picks")

#Show the plot
plt.show()

dfgames = pd.DataFrame(games)
HomeScores = list(dfgames.iloc[:, 10])
AwayScores = list(dfgames.iloc[:, 8])
HomeWins = 0
HomeLoss = 0
HomeTie = 0

for row in range(len(dfgames)):
    if HomeScores[row] != None and AwayScores[row] != None:
        if HomeScores[row] > AwayScores[row]:
            HomeWins += 1
        elif HomeScores[row] < AwayScores[row]:
            HomeLoss += 1
        else:
            HomeTie += 1

print(HomeWins)
print(HomeLoss)
print(HomeTie)

AggHomeWins = HomeWins/len(dfgames)
AggHomeLoss = HomeLoss/len(dfgames)
AggHomeTie = HomeTie/len(dfgames)
                      
print(AggHomeWins)
print(AggHomeLoss)
print(AggHomeTie)

aggregate = [AggHomeWins,AggHomeLoss,AggHomeTie]
labels = 'Home Team Wins','Home Team Losses', 'Home Team Ties'

plt.figure(figsize=(4,4))
plt.pie(aggregate, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

#creating data for winning record
nfl2021['winningRecord'] = nfl2021['Wins'] > nfl2021['Losses']
week13data['winningRecord'] = week13data['Wins'] > week13data['Losses']
nfl2020['winningRecord'] = nfl2020['Wins'] > nfl2020['Losses']
nfl2019['winningRecord'] = nfl2019['Wins'] > nfl2019['Losses']
nfl2018['winningRecord'] = nfl2018['Wins'] > nfl2018['Losses']

nfl2018.head()

#creating training set
data_frames = [nfl2020,nfl2019,nfl2018] #combining previous seasons
stat_total = pd.concat(data_frames) #cleaning up data
stat_total['Year'] = stat_total['Year'].astype('int')
print(stat_total)
trainSet = stat_total

#creating test set for week 8
nfl2021['Year'] = nfl2021['Year'].astype('int')
testSetweek8 = nfl2021
testSetweek8.head()

#creating dataset for week 13
week13data['Year'] = week13data['Year'].astype('int')
testSetweek13 = week13data
testSetweek13.head()

#combining training and test data
mydataweek8 = pd.concat([testSetweek8, trainSet])
mydataweek8

#combining training and test data
mydataweek13 = pd.concat([testSetweek13, trainSet])
mydataweek13

#setting up our training and testing variables
train_columns = ['Points Scored/Game','Points Allowed/Game','Point Diffferntial','Percentage of Scores','Percentage of Turnovers','Third Down Conversion','Red Zone Conversion']
X_week8 = np.asarray(mydataweek8[train_columns])
Y_superbowl8 = np.asarray(mydataweek8['Super Bowl'])
Y_playoffs8 = np.asarray(mydataweek8['Playoffs'])
Y_winning8 = np.asarray(mydataweek8['winningRecord'])
X_week13 = np.asarray(mydataweek13[train_columns])
Y_superbowl13 = np.asarray(mydataweek13['Super Bowl'])
Y_playoffs13 = np.asarray(mydataweek13['Playoffs'])
Y_winning13 = np.asarray(mydataweek13['winningRecord'])
print(X_week8.shape)
print(Y_superbowl8.shape)
print(Y_playoffs8.shape)
print(Y_winning8.shape)
print(X_week13.shape)
print(Y_superbowl13.shape)
print(Y_playoffs13.shape)
print(Y_winning13.shape)

teams = ['Arizona Cardinals','Atlanta Falcons','Baltimore Ravens','Buffalo Bills',
         'Carolina Panthers','Chicago Bears','Cincinnati Bengals','Cleveland Browns','Dallas Cowboys','Denver Broncos',
        'Detroit Lions','Green Bay Packers','Houston Texans','Indianapolis Colts','Jacksonville Jaguars','Kansas City Chiefs',
        'Las Vegas Raiders','Los Angeles Chargers','Los Angeles Rams','Miami Dolphins','Minnesota Vikings',
         'New England Patriots','New Orleans Saints','New York Giants','New York Jets','Philadelphia Eagles',
        'Pittsburgh Steelers','San Francisco 49ers','Seattle Seahawks','Tampa Bay Buccaneers','Tennessee Titans',
        'Washington Football Team'] #keep team names constant

#spilting test and training data
X_test8 = X_week8[0:32]
X_train8 = X_week8[32:]

Y_testSB8 = Y_superbowl8[0:32]
Y_trainSB8 = Y_superbowl8[32:]

Y_testPlayoffs8 = Y_playoffs8[0:32]
Y_trainPlayoffs8 = Y_playoffs8[32:]

Y_testWinning8 = Y_winning8[0:32]
Y_trainWinning8 = Y_winning8[32:]

X_test13 = X_week13[0:32]
X_train13 = X_week13[32:]

Y_testSB13 = Y_superbowl13[0:32]
Y_trainSB13 = Y_superbowl13[32:]

Y_testPlayoffs13 = Y_playoffs13[0:32]
Y_trainPlayoffs13 = Y_playoffs13[32:]

Y_testWinning13 = Y_winning13[0:32]
Y_trainWinning13 = Y_winning13[32:]

print(X_test8)
print(Y_testSB8)
print(Y_testPlayoffs8)
print(Y_testWinning8)
print(X_train8)
print(Y_trainSB8)
print(Y_trainPlayoffs8)
print(Y_trainWinning8)

#Fit the 2021 data to the logisitc regression model
#First do the Super Bowl odds
week8SBLR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train8,Y_trainSB8)

week13SBLR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train13,Y_trainSB13)

#Predict SB favorites with model
y_predSB8 = week8SBLR.predict(X_test8)
y_predSB8

#Predict SB favorites with model
y_predSB13 = week13SBLR.predict(X_test13)
y_predSB13

#Getting Probability estimates for each team
ypredSB8_prob = week8SBLR.predict_proba(X_test8)
ypredSB8_prob

#Getting Probability estimates for each team
ypredSB13_prob = week13SBLR.predict_proba(X_test13)
ypredSB13_prob

#displays the Superbowl odds
superbowloddsweek8 = pd.DataFrame({'Team':teams, 'Prediction':ypredSB8_prob[:,1]}) 

superbowloddsweek8 = superbowloddsweek8.sort_values(by = ['Prediction'], ascending = False)
superbowloddsweek8['Prediction_Rank'] = superbowloddsweek8['Prediction'].rank(ascending = False)

superbowloddsweek8

superbowloddsweek13 = pd.DataFrame({'Team':teams, 'Prediction':ypredSB13_prob[:,1]}) 

superbowloddsweek13 = superbowloddsweek8.sort_values(by = ['Prediction'], ascending = False)
superbowloddsweek13['Prediction_Rank'] = superbowloddsweek13['Prediction'].rank(ascending = False)

superbowloddsweek13

#Fit the 2021 data to the logisitc regression model
#Playoff odds
week8PLR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train8,Y_trainPlayoffs8)

week13PLR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train13,Y_trainPlayoffs13)

#Predict playoff odds with model
y_predP8 = week8PLR.predict(X_test8)
y_predP8

#Predict playoff odds with model
y_predP13 = week13PLR.predict(X_test13)
y_predP13

#Getting Probability estimates for each team
ypredP8_prob = week8PLR.predict_proba(X_test8)
ypredP8_prob

#Getting Probability estimates for each team
ypredP13_prob = week13PLR.predict_proba(X_test13)
ypredP13_prob

#displays the Playoff odds
playoffoddsweek8 = pd.DataFrame({'Team':teams, 'Prediction':ypredP8_prob[:,1]}) 

playoffoddsweek8 = playoffoddsweek8.sort_values(by = ['Prediction'], ascending = False)
playoffoddsweek8['Prediction_Rank'] = playoffoddsweek8['Prediction'].rank(ascending = False)

playoffoddsweek8

#displays the Playoff odds
playoffoddsweek13 = pd.DataFrame({'Team':teams, 'Prediction':ypredP13_prob[:,1]}) 

playoffoddsweek13 = playoffoddsweek13.sort_values(by = ['Prediction'], ascending = False)
playoffoddsweek13['Prediction_Rank'] = playoffoddsweek13['Prediction'].rank(ascending = False)

playoffoddsweek13

week8WLR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train8,Y_trainWinning8)

week13WLR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train13,Y_trainWinning13)

#Predict winning record with model
y_predW8 = week8WLR.predict(X_test8)
y_predW8

#Predict winning record with model
y_predW13 = week13WLR.predict(X_test13)
y_predW13

#Getting Probability estimates for each team
ypredW8_prob = week8WLR.predict_proba(X_test8)
ypredW8_prob

#Getting Probability estimates for each team
ypredW13_prob = week13WLR.predict_proba(X_test13)
ypredW13_prob

#displays the Playoff odds
winningrecordweek8 = pd.DataFrame({'Team':teams, 'Prediction':ypredW8_prob[:,1]}) 

winningrecordweek8 = winningrecordweek8.sort_values(by = ['Prediction'], ascending = False)
winningrecordweek8['Prediction_Rank'] = winningrecordweek8['Prediction'].rank(ascending = False)

winningrecordweek8

#displays the Playoff odds
winningrecordweek13 = pd.DataFrame({'Team':teams, 'Prediction':ypredW13_prob[:,1]}) 

winningrecordweek13 = winningrecordweek13.sort_values(by = ['Prediction'], ascending = False)
winningrecordweek13['Prediction_Rank'] = winningrecordweek13['Prediction'].rank(ascending = False)

winningrecordweek13

pointdiff2020 = nfl2020.sort_values(
    by=["Point Diffferntial"],
    ascending=[False]
)[["Point Diffferntial"]]

best_off2020 = nfl2020.sort_values(
    by=["Points Scored/Game"],
    ascending=[False]
)[["Points Scored/Game"]]

best_def2020 = nfl2020.sort_values(
    by=["Points Allowed/Game"],
    ascending=[False]
)[["Points Allowed/Game"]]

best_on_third_down2020 = nfl2020.sort_values(
    by=["Third Down Conversion"],
    ascending=[False]
)[["Third Down Conversion"]]

redzone_conversion_percentage2020 = nfl2020.sort_values(
    by=["Red Zone Conversion"],
    ascending=[False]
)[["Red Zone Conversion"]]

# Best in each category
print(pointdiff2020)
print(best_off2020)
print(best_def2020)
print(best_on_third_down2020)
print(redzone_conversion_percentage2020)

pointdiff2021 = nfl2021.sort_values(
    by=["Point Diffferntial"],
    ascending=[False]
)[["Point Diffferntial"]]

best_off2021 = nfl2021.sort_values(
    by=["Points Scored/Game"],
    ascending=[False]
)[["Points Scored/Game"]]

best_def2021 = nfl2021.sort_values(
    by=["Points Allowed/Game"],
    ascending=[False]
)[["Points Allowed/Game"]]

best_on_third_down2021 = nfl2021.sort_values(
    by=["Third Down Conversion"],
    ascending=[False]
)[["Third Down Conversion"]]

redzone_conversion_percentage2021 = nfl2021.sort_values(
    by=["Red Zone Conversion"],
    ascending=[False]
)[["Red Zone Conversion"]]

# Best in each category
print(pointdiff2021)
print(best_off2021)
print(best_def2021)
print(best_on_third_down2021)
print(redzone_conversion_percentage2021)