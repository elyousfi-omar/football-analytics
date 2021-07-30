# Libraries
from tkinter import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from matplotlib.patches import Arc
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk,Image
from pandastable import Table, TableModel
from tkinter import ttk
from ttkbootstrap import Style
import os
from plotly import graph_objects, offline
from matplotlib.backends.backend_pdf import PdfPages
from pfe import createPitch, Field, add_possession_chain_sb, createGoalMouth
from tkinter import messagebox
import ast
from tkinter import filedialog
import sys 
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------------------------#

# Set up the main window
root = Tk()
root.title('Football Analytics v1.')

# Window geometry
root.geometry("720x320+430+220")

# Fixed window size
root.minsize(720, 320)
root.maxsize(720, 320)

# Favicon
root.iconbitmap('favicon.ico')
np.seterr(divide='ignore', invalid='ignore')

# The fixed pitch width
pitchLengthX=120
pitchWidthY=80

# Plots styling
plt.style.use('ggplot')

# -------------------------------------------------------------------------#
#### --- Teams Analysis --- ####

# Goal scorers list
def goalScorers():
	sco = Toplevel(root)
	season = var.get()
	av = []
	available = os.listdir("scorers")

	for a in available:
		av.append(a.strip(".csv"))
	if team_lookup.get() in av:
		scorers = 	pd.read_csv("scorers\\" + team_lookup.get() + ".csv", index_col=0)

		f = Frame(sco)
		f.grid(row=0, column=0)
		df = pd.DataFrame(scorers[scorers["Season"] == season].value_counts()).reset_index()
		table = pt = Table(f, dataframe=df,showtoolbar=True, showstatusbar=True)
		pt.show()
	else:
		messagebox.showinfo("Not available", "The scorers dataset for "+ team_lookup.get() +" isn't available yet..")
		sco.destroy()

# Shots analysis
def AnalyzeShots():
	analyze = Toplevel(root)
	analyze.title("Shots analysis")
	analyze.iconbitmap('favicon.ico')
	global shots_df
	global H_Shot
	global H_Goal
	global goals_only
	season = var.get()

	title = ttk.Label(analyze, text = team_lookup.get() + "'s shots analysis in " + season, font=("Helvatica", 14))
	title.grid(row=0, column=0)
	
	av = []
	available = os.listdir("shots_df")
	for a in available:
		av.append(a.strip(".csv"))
	
	if team_lookup.get() in av:
		shots_df = 	pd.read_csv("shots_df\\" + team_lookup.get() + ".csv")
		if season != "All":
			shots_df = shots_df[shots_df["Season"] == season]
		elif season == "All":
			pass
		# Convert Location column from string type to list type
		shots_df.Location = shots_df.Location.apply(lambda s: list(ast.literal_eval(s)))

		# Split the location column to X and Y coordinates
		for i,shot in shots_df.iterrows():
			shots_df.at[i,"X"] = shot["Location"][0]
			shots_df.at[i,"Y"] = shot["Location"][1]

		# Create a Goal column, 1 == GOAL, 0 == NOT A GOAL
		for i,shot in shots_df.iterrows():
			if shot["Shot_Outcome"] == "Goal":
				shots_df.at[i,"Goal"] = 1
			else:
				shots_df.at[i,"Goal"] = 0

		# Create Distance and Angle columns (Explained in the project report)
		for i,shot in shots_df.iterrows():

			shots_df.at[i,'X'] = 120-shot["X"]
			x = shots_df.at[i,'X'] 
			y = abs(shots_df.at[i,'Y'] - 40)

			shots_df.at[i,'Distance']=np.sqrt(x**2 + y**2)
			
			a = np.arctan(8 * x /(x**2 + y**2 - (8/2)**2))
			if a<0:
				a=np.pi+a
			shots_df.at[i,'Angle'] =a

		#Two dimensional histogram
		H_Shot=np.histogram2d(shots_df['X'], shots_df['Y'],bins=50,range=[[0, 120],[0, 80]])
		goals_only=shots_df[shots_df['Goal']==1]
		H_Goal=np.histogram2d(goals_only['X'], goals_only['Y'],bins=50,range=[[0, 120],[0, 80]])

		quadro = ttk.LabelFrame(analyze, text="Overall:")
		quadro.grid(row=2, column=0, padx=2)

		# Display all shots
		display = ttk.Button(quadro, text="Shots Heatmap", command = displayShots)
		display.grid(row=0, column=0, pady=2, padx=2, ipadx=18)

		# Display only scored shots
		scored = ttk.Button(quadro, text="Goals Heatmap", command = scoredShots)
		scored.grid(row=1, column=0, pady=2, padx=2, ipadx=5)

		# Frame
		predictions = ttk.LabelFrame(analyze, text = "Expected Goals:")
		predictions.grid(row=1, column=0, padx=2)

		# Proportion of a shot resulting in a goal
		propor = ttk.Button(predictions, text="Proportion of shots resulting in a goal", command = proportion)
		propor.grid(row=0, column=0, columnspan=3, pady=2, ipadx=14)

		# Expected goals metric (Gradient Boosting classifier)
		proba = ttk.Button(predictions, text="Expected goals", command = proba_scored)
		proba.grid(row=1, column=0, columnspan=3, pady=2, ipadx=14)

		# Frame
		titre = ttk.Label(predictions, text="Probablity a chance scored by:")
		titre.grid(row=2, column=0, pady=2, padx=2)

		# Probability a shot scored
		# Angle
		angleshot = ttk.Button(predictions, text="Shot angle", command = shotAngle)
		angleshot.grid(row=3, column=0, pady=2, padx=2, ipadx=40)

		# Distance
		dis = ttk.Button(predictions, text="Shot distance from the target", command = distanceChance)
		dis.grid(row=3, column=1, pady=2, padx=5)

	else:
		# Error message if the team data isn't in the repository
		messagebox.showinfo("Not available", "The shots dataset for "+ team_lookup.get() +" isn't available yet..")
		analyze.destroy()

# Functions of the shots analysis
def proba_scored():
	label_encoder = LabelEncoder()
	shots_df["Body part"] = label_encoder.fit_transform(shots_df["Body part"])
	shots_df["Shot type"] = label_encoder.fit_transform(shots_df["Shot type"])
	shots_df["Player"] = label_encoder.fit_transform(shots_df["Player"])
	X = shots_df[["Distance", "Angle", "Body part", "Shot type", 'Player']]
	y = shots_df["Goal"]
	
	model = pickle.load(open('final_model_x', 'rb'))
	proba = model.predict_proba(X)
	fig, ax = plt.subplots(1)
	ax = sns.scatterplot(x="Distance", y="Angle", hue=proba[:,1], data=shots_df)
	fig.set_size_inches(10,8)
	plt.show()

def displayShots():
	(fig,ax) = createGoalMouth()
	pos=ax.imshow(H_Shot[0], extent=[-1,66,104,-1], aspect='auto',cmap=plt.cm.Reds)
	fig.colorbar(pos, ax=ax)
	ax.set_title('Number of shots')
	plt.xlim((-1,66))
	plt.ylim((-3,35))
	plt.tight_layout()
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()

def scoredShots():
	(fig,ax) = createGoalMouth()
	pos=ax.imshow(H_Goal[0], extent=[-1,66,104,-1], aspect='auto',cmap=plt.cm.Reds)
	fig.colorbar(pos, ax=ax)
	ax.set_title('Scored Shots')
	plt.xlim((-1,66))
	plt.ylim((-3,35))
	plt.tight_layout()
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()

def proportion():
	(fig,ax) = createGoalMouth()
	pos=ax.imshow(H_Goal[0]/H_Shot[0], extent=[-1,66,104,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=0.5)
	fig.colorbar(pos, ax=ax)
	ax.set_title('Proportion of shots resulting in a goal')
	plt.xlim((-1,66))
	plt.ylim((-3,35))
	plt.tight_layout()
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()
	
def distanceChance():
	shotcount_dist=np.histogram(shots_df['Distance'],bins=40,range=[0, 70])
	goalcount_dist=np.histogram(goals_only['Distance'],bins=40,range=[0, 70])
	prob_goal=np.divide(goalcount_dist[0],shotcount_dist[0])
	distance=shotcount_dist[1]
	middistance= (distance[:-1] + distance[1:])/2
	fig,ax=plt.subplots(num=1)
	ax.plot(middistance, prob_goal, linestyle='none', marker= '.', color='black')
	ax.set_ylabel('Probability chance scored')
	ax.set_xlabel("Distance from target")
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	plt.show()

def shotAngle():
	shotcount_dist=np.histogram(shots_df['Angle']*180/np.pi,bins=40,range=[0, 150])
	goalcount_dist=np.histogram(goals_only['Angle']*180/np.pi,bins=40,range=[0, 150])
	prob_goal=np.divide(goalcount_dist[0],shotcount_dist[0])
	angle=shotcount_dist[1]
	midangle= (angle[:-1] + angle[1:])/2
	fig,ax=plt.subplots(num=2)
	ax.plot(midangle, prob_goal, linestyle='none', marker= '.', markersize= 12, color='black')
	ax.set_ylabel('Probability chance scored')
	ax.set_xlabel("Shot angle")
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	plt.show()

# Team analysis (results)
def analyzeTeam():
	ta = Toplevel(root)
	ta.title("Results")
	ta.geometry("400x600+900+50")
	ta.iconbitmap('favicon.ico')

	main_frame = Frame(ta)
	main_frame.pack(fill=BOTH, expand=1)

	my_canvas = Canvas(main_frame)
	my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

	my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
	my_scrollbar.pack(side=RIGHT, fill=Y)

	my_canvas.configure(yscrollcommand=my_scrollbar.set)
	my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all")))

	second_frame = Frame(my_canvas)
	my_canvas.create_window((0,0), window=second_frame, anchor="nw")

	competitions_id = [2,11,16,37,43,49,72]
	global wins
	global loss
	global draws
	global nbr_goals
	global nbr_games

	wins = loss = draws = nbr_goals = 0
	season = var.get()
	for competi in competitions_id:
		files = [f for f in os.listdir('matches\\' + str(competi) )]
		for file in files:
			with open('matches\\'+ str(competi) + "/" + file, encoding='utf-8') as f:
				temp = json.load(f)
				for match in temp:
					home_team_name=match['home_team']['home_team_name']
					away_team_name=match['away_team']['away_team_name']
					competition_name_home = match['competition']['competition_name']
					seas = match["season"]["season_name"]
					if seas == season:
						if (home_team_name==team_lookup.get()) or (away_team_name==team_lookup.get()):
							home_score = match['home_score']
							away_score = match['away_score']
							describe_text = home_team_name + ' VS ' + away_team_name
							result_text = ' finished ' + str(home_score) +  ' : ' + str(away_score)
							if(home_team_name == team_lookup.get()):
								if(home_score>away_score):
									wins += 1
								elif(home_score == away_score):
									draws += 1
								else:
									loss += 1
								nbr_goals = nbr_goals + home_score
							elif(away_team_name == team_lookup.get()):
								if(away_score>home_score):
									wins += 1
								elif(home_score == away_score):
									draws += 1
								else:
									loss += 1
								nbr_goals = nbr_goals + away_score
							

							ttk.Label(second_frame, text= describe_text + result_text + " In " + competition_name_home).pack()

	ttk.Separator(second_frame,orient=HORIZONTAL).pack(expand=True, fill="x")
	nbr_games = wins + draws + loss
	ttk.Label(second_frame, text= "Number of Games of " + team_lookup.get() + " : " + str(nbr_games)).pack()
	ttk.Label(second_frame, text= "Number of wins of " + team_lookup.get() + " : " + str(wins)).pack()
	ttk.Label(second_frame, text= "Number of draws of " + team_lookup.get() + " : " + str(draws)).pack()
	ttk.Label(second_frame, text= "Number of losses of " + team_lookup.get() + " : " + str(loss)).pack()
	ttk.Label(second_frame, text= "Number of goals scored by " + team_lookup.get() + " : " + str(nbr_goals)).pack()

	#won = ttk.Button(second_frame, text="Wins", command = wonMatches).pack()
	#lost = ttk.Button(second_frame, text="Losses", command = lostMatches).pack()
	#drew = ttk.Button(second_frame, text="Draws", command = DrawnMatches).pack()

	# The plot code:
	results = {"Wins": wins, "Draws": draws, "Losses": loss}
	c, result_plot = plt.subplots(figsize = (8,5))
	result_plot.barh(range(len(results)), list(results.values()), tick_label = list(results.keys()), color=['green','grey','#E24A33'])
	for i in result_plot.patches:
		plt.text(i.get_width()+0.2, i.get_y()+0.5, str(round((i.get_width()), 2)), fontsize = 10, fontweight ='bold', color ='grey')
	plt.show()

# Manager of the team in a certain season
def teamInfo():
	ex = Toplevel(root)
	ex.title("Manager")
	ex.geometry("300x100")
	ex.iconbitmap('favicon.ico')
	competitions_id = [2,11,16,37,43,49,72]
	managers = []
	home_score = away_score = 0
	wins = loss = 0
	
	for competi in competitions_id:
		files = [f for f in os.listdir('matches\\' + str(competi) )]
		for file in files:
			with open('matches\\'+ str(competi) + "/" + file, encoding='utf-8') as f:
				temp = json.load(f)
			for match in temp:
				home_team_name=match['home_team']['home_team_name']
				away_team_name=match['away_team']['away_team_name']
				if('managers' in match['home_team'] and match['season']['season_name'] == var.get()):
					home_manager = match['home_team']['managers'][0]['name']
					away_manager = match['away_team']['managers'][0]['name']
					if (home_team_name == team_lookup.get()) or (away_team_name == team_lookup.get()):
						if(home_team_name == team_lookup.get()):
							managers.append(home_manager)
						elif(away_team_name == team_lookup.get()):
							managers.append(away_manager)

	ttk.Label(ex,text = "Manager of " + str(team_lookup.get()) + " In " + str(var.get())).pack()
	for i in list(dict.fromkeys(managers)):
		ttk.Label(ex,text = str(i)).pack()

# Won matches
def wonMatches():
	won = Toplevel(root)

	competitions_id = [2,11,16,37,43,49,72]
	for competi in competitions_id:
		files = [f for f in os.listdir('matches\\' + str(competi) )]
		for file in files:
			with open('matches\\'+ str(competi) + "/" + file, encoding='utf-8') as f:
				temp = json.load(f)
			temp = temp[temp['season']['season_name'] == var.get()]
			for match in temp:
				home_team_name=match['home_team']['home_team_name']
				away_team_name=match['away_team']['away_team_name']
				competition_name_home = match['competition']['competition_name']
				#mmid = temp[i]["match_id"]
				if (home_team_name==team_lookup.get()) or (away_team_name==team_lookup.get()):
					home_score = match['home_score']
					away_score = match['away_score']
					describe_text = home_team_name + ' VS ' + away_team_name
					result_text = ' finished ' + str(home_score) +  ' : ' + str(away_score)
					if(home_team_name == team_lookup.get()):
						if(home_score>away_score):
							ttk.Label(won, text= describe_text + result_text + " In " + competition_name_home).pack()
					elif(away_team_name == team_lookup.get()):
						if(away_score>home_score):
							ttk.Label(won, text= describe_text + result_text + " In " + competition_name_home).pack()

# Drawn matches						
def DrawnMatches():
	d = Toplevel(root)

	competitions_id = [2,11,16,37,43,49,72]

	for competi in competitions_id:
		files = [f for f in os.listdir('matches\\' + str(competi) )]
		for file in files:
			with open('matches\\'+ str(competi) + "/" + file, encoding='utf-8') as f:
				temp = json.load(f)
				for match in temp:
					home_team_name=match['home_team']['home_team_name']
					away_team_name=match['away_team']['away_team_name']
					competition_name_home = match['competition']['competition_name']
					#mmid = temp[i]["match_id"]
					if (home_team_name==team_lookup.get()) or (away_team_name==team_lookup.get()):
						home_score = match['home_score']
						away_score = match['away_score']
						describe_text = home_team_name + ' VS ' + away_team_name
						result_text = ' finished ' + str(home_score) +  ' : ' + str(away_score)
						if(home_team_name == team_lookup.get()):
							if(home_score == away_score):
								ttk.Label(d, text= describe_text + result_text + " In " + competition_name_home).pack()	
						elif(away_team_name == team_lookup.get()):
							if(home_score == away_score):
								ttk.Label(d, text= describe_text + result_text + " In " + competition_name_home).pack()	

						
# Lost matches
def lostMatches():
	l = Toplevel(root)

	competitions_id = [2,11,16,37,43,49,72]

	for competi in competitions_id:
		files = [f for f in os.listdir('matches\\' + str(competi) )]
		for file in files:
			with open('matches\\'+ str(competi) + "/" + file, encoding='utf-8') as f:
				temp = json.load(f)
				for match in temp:
					home_team_name=match['home_team']['home_team_name']
					away_team_name=match['away_team']['away_team_name']
					competition_name_home = match['competition']['competition_name']
					#mmid = temp[i]["match_id"]
					if (home_team_name==team_lookup.get()) or (away_team_name==team_lookup.get()):
						home_score = match['home_score']
						away_score = match['away_score']
						describe_text = home_team_name + ' VS ' + away_team_name
						result_text = ' finished ' + str(home_score) +  ' : ' + str(away_score)
						if(home_team_name == team_lookup.get()):
							if(home_score<away_score):
								ttk.Label(l, text= describe_text + result_text + " In " + competition_name_home).pack()		

						elif(away_team_name == team_lookup.get()):
							if(away_score<home_score):
								ttk.Label(l, text= describe_text + result_text + " In " + competition_name_home).pack()		

						

# Consult the ID/MATCH file						
def teamslist():
	os.startfile("matches_guide.pdf")

# Change the theme of the interface
def gettheme():
	if tkvar.get() == "Light Theme":
		style = Style(theme="sandstone", themes_file = "ttkbootstrap_themes.json")
	elif tkvar.get() == "Dark Theme":
		style = Style(theme="darktheme", themes_file = "ttkbootstrap_themes.json")
	window = style.master

# Number of matches for each team in the dataset
def nbrMatches():
	os.startfile("team_nbrgames.pdf")

# Competitions details
def compeData():
	competitions_data = Toplevel(root)
	pd.set_option('display.max_columns', None)
	with open('competitions.json', encoding='utf-8') as f:
		competitions = json.load(f)
	df_c = pd.json_normalize(competitions, sep = "_")
	df_c = df_c.set_index(["competition_id","season_id"])
	df_c = df_c.sort_values("competition_id")
	os.startfile("competitions.pdf")

######### Match Analysis ########
# Summary of the match
def summary():
	suma = Toplevel(matchanalysis)
	
	zero = ttk.Label(suma)
	zero.grid(row=0, column=0)
	res = ttk.Label(zero, text=home_team + ' ' +str(matches['home_score']) + '-' + str(matches['away_score']) + ' ' + away_team, font=('Helvatica', 12, 'bold'))
	res.grid(row=0, column=0)

	first = ttk.LabelFrame(suma, text="Scorers")
	first.grid(row=1, column=0, padx=2)
	goals = df[(df["shot_outcome_name"] == 'Goal') | (df['type_name'] == 'Own Goal Against')]
	for i, row in goals.iterrows():
		result = ttk.Label(first, text=row['player_name'] + ' - ' +  row['possession_team_name'] + ' - ' +str(row['minute']) + ':' + str(row['second']))
		result.grid(row=i, column=0)

	second = ttk.LabelFrame(suma, text="Substitutions")
	second.grid(row=2, column=0)
	subst = df[df['type_name'] == 'Substitution']
	for i, row in subst.iterrows():
		subs = ttk.Label(second, text='Out: ' + row['player_name'] + ' | ' + 'In: ' + row['substitution_replacement_name'] + ' | Reason: ' + row['substitution_outcome_name'])
		subs.grid(row=i, column=0)

# Lineups
def showTeams():
	lineup = Toplevel(matchanalysis)
	lineup.title("Lineup")
	lineup.geometry("600x600")
	lineup.iconbitmap('favicon.ico')

	formation_home = ttk.Label(lineup, text= int(df['tactics_formation'][0]))
	formation_away = ttk.Label(lineup, text= int(df['tactics_formation'][1]))
	

	head0 = ttk.Label(lineup, text="Name    ||    Position    ||    Jersey number")

	
	hometeam = ttk.Label(lineup, text = home_team + " Lineup:", font=("Helvatica", 15 ,"bold"))
	hometeam.pack()
	formation_home.pack()

	head0.pack()

	for i in range(11):
		home = ttk.Label(lineup, text= df['tactics_lineup'][0][i].get("player").get("name") + "  " + df['tactics_lineup'][0][i].get("position").get("name") + "  " + str(df['tactics_lineup'][0][i].get("jersey_number")))
		home.pack()
	
	head1 = ttk.Label(lineup, text="Name    ||    Position    ||    Jersey number")
	
	awayteam = ttk.Label(lineup, text = away_team + " Lineup:", font=("Helvatica", 15 ,"bold"))
	awayteam.pack()
	formation_away.pack()
	head1.pack()

	for j in range(11):
		away = ttk.Label(lineup, text = df['tactics_lineup'][1][j].get("player").get("name") + "  " + df['tactics_lineup'][1][j].get("position").get("name") + "  " + str(df['tactics_lineup'][1][j].get("jersey_number")))
		away.pack()

# Passes Ranking
def showMostPasses():
	mostpasses = Toplevel(matchanalysis)
	mostpasses.title('Passes Ranking')
	mostpasses.geometry('800x600')
	mostpasses.iconbitmap('favicon.ico')

	rankingpasses = pd.DataFrame(df.loc[df['type_name'] == 'Pass'].set_index('id'))
	def passVisualization():
		countplt, ax = plt.subplots(figsize = (8,5))
		ax = sns.countplot(y=rankingpasses["player_name"], order=rankingpasses["player_name"].value_counts().iloc[:-10].index)
		ax.set_title('Passes Ranking',fontsize = 18, fontweight='bold' )
		ax.set_xlabel('Number of passes', fontsize = 15)
		ax.set_ylabel('Players', fontsize = 15)
		plt.show()

	d = rankingpasses[['player_name','team_name']].value_counts()
	f = Frame(mostpasses)
	f.grid(row=0, column=0)
	table = pt = Table(f, dataframe=pd.DataFrame(d).reset_index(), showtoolbar=True, showstatusbar=True, editable=False)
	pt.columncolors['player_name'] = '#dcf1fc' #color a specific column
	pt.redraw()
	pt.show()

	ttk.Button(mostpasses, text="Visualize it!", command=passVisualization).grid(row=1, column=0)

# Shots 
def shots():
	shots = df.loc[df['type_name'] == 'Shot'].set_index('id')
	sns.countplot(y=shots["player_name"])
	plt.show()

# Possession chain
def possAnalysis():
	relevant_possessions = df[(df['type_id'] == 16)]['possession'].unique() 
	ground = Field()
	ground.add_title('Possession chains')
	for possession in relevant_possessions:

		p = df[df['possession'] == possession] # get all events of the possession
		add_possession_chain_sb(ground.figure, p, team_name= df.iloc[1]['team_name'])

	ground.save('possessions_' + lookup.get() + '.html')
	os.startfile('possessions_' + lookup.get() + '.html')

# Dribbles of both teams
def analyzeDribbles():
	dribbles = df.loc[df['type_name']=="Dribble"].set_index("id")
	(fig,ax) = createPitch(120,80,'yards','gray')
	home_team = df['team_name'][0]
	away_team = df['team_name'][1]
	for i,dribble in dribbles.iterrows():
		x=dribble['location'][0]
		y=dribble['location'][1]

		success = dribble['dribble_outcome_name'] == 'Complete'
		team_name = dribble['team_name']

		circleSize=2

		if (team_name==home_team):
			if success:
				dribbleCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="red")
				plt.text((x+1),pitchWidthY-y+1,dribble['player_name']) 
			else:
				dribbleCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="red")     
				dribbleCircle.set_alpha(.2)
		elif (team_name==away_team):
			if success:
				dribbleCircle=plt.Circle((pitchLengthX-x,y),circleSize,color="blue") 
				plt.text((pitchLengthX-x+1),y+1,dribble['player_name']) 
			else:
				dribbleCircle=plt.Circle((pitchLengthX-x,y),circleSize,color="blue")      
				dribbleCircle.set_alpha(.2)
		ax.add_patch(dribbleCircle)

	fig.set_size_inches(10, 7)
	plt.show()

# Shots locations
def shots_locations():
	shots = df.loc[df['type_name'] == 'Shot'].set_index('id')
	(fig,ax) = createPitch(120,80,'yards','gray')
	for i,shot in shots.iterrows():
		x=shot['location'][0]
		y=shot['location'][1]

		goal=shot['shot_outcome_name']=='Goal'
		team_name = shot['team_name']

		circleSize=2

		if (team_name ==home_team ):
			if goal:
				shotCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="red")
				plt.text((x+1),pitchWidthY-y+1,shot['player_name']) 
			else:
				shotCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="red")     
				shotCircle.set_alpha(.2)
		elif (team_name==away_team):
			if goal:
				shotCircle=plt.Circle((pitchLengthX-x,y),circleSize,color="blue") 
				plt.text((pitchLengthX-x+1),y+1,shot['player_name']) 
			else:
				shotCircle=plt.Circle((pitchLengthX-x,y),circleSize,color="blue")      
				shotCircle.set_alpha(.2)
		ax.add_patch(shotCircle)

	owngoals = df.loc[df['type_name'] == 'Own Goal Against'].set_index('id')
	for i, og in owngoals.iterrows():
		x = og['location'][0]
		y = og['location'][1]

		team_name = og['team_name']
		if (team_name == df['team_name'][0] ):
			shotCircle=plt.Circle((x,80-y),circleSize,color="yellow")
			plt.text((x+1),80-y+1,og['player_name']) 
		elif (team_name== df['team_name'][1]):
			shotCircle=plt.Circle((120-x,y),circleSize,color="green") 
			plt.text((120-x+1),y+1,og['player_name']) 

		ax.add_patch(shotCircle)
	fig.set_size_inches(10, 7)
	plt.show()

# Fouls locations
def fouls():
	fouls = df.loc[df['type_name'] == 'Foul Committed'].set_index('id')
	(fig,ax) = createPitch(120,80,'yards','gray')
	for i,foul in fouls.iterrows():
	    x=foul['location'][0]
	    y=foul['location'][1]
	    
	    #goal=foul['shot_outcome_name']=='Goal'
	    team_name=foul['team_name']
	    
	    circleSize=2
	    #circleSize=np.sqrt(shot['shot_statsbomb_xg'])*12

	    if (team_name==home_team):
	        shotCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="red")     
	        shotCircle.set_alpha(.2)
	    elif (team_name==away_team):
	        shotCircle=plt.Circle((pitchLengthX-x,y),circleSize,color="blue")      
	        shotCircle.set_alpha(.2)
	    ax.add_patch(shotCircle)

	fig.set_size_inches(10, 7)
	plt.show()

# Cards issued to players
def tackles():
	cards = Toplevel(matchanalysis)
	cards.title('Cards')
	cards.geometry('400x200')
	cards.iconbitmap('favicon.ico')

	c = str(df[['foul_committed_card_name','player_name','team_name']].value_counts()).rsplit(' ',2)[0]
	h = ttk.Label(cards, text= "Type      ||     Player     ||     Team", font=("Helvatica",10,"bold"))

	h.pack()
	cardsgiven = ttk.Label(cards, text= c[55:])
	cardsgiven.pack()

# Missed passes plot
def missedPassesPlot():
	missedpass = Toplevel(root)
	missedpass.title('Missed Passes')
	missedpass.geometry('600x800')
	missedpass.iconbitmap('favicon.ico')
	global missed

	# Select missed passes
	missed = df[df['pass_outcome_name'].isin(["Incomplete","Out","Pass Offside", "Unknown"])]

	# number of missed passes by players
	m = pd.DataFrame(missed["player_name"].value_counts())
	m.columns = [""]

	# number of passes by players
	passes = df.loc[df['type_name'] == 'Pass'].set_index('id')
	p = pd.DataFrame(passes["player_name"].value_counts())
	p.columns = [""]

	# create a new columns named frequency and sort the values in the descending order
	res = pd.concat([p,m], axis=1, keys=["Passes made", "Missed passes"])
	res["Frequency"] = (((res["Missed passes"]/res["Passes made"])*100).round(1).astype(str)+"%")
	res = res.sort_values("Frequency", ascending=False)
	res = pd.DataFrame(res).reset_index()

	f = Frame(missedpass)
	f.grid(row=0, column=0)
	table = pt = Table(f, dataframe=res ,showtoolbar=True, showstatusbar=True, editable=False)
	pt.show()
	
	ttk.Button(missedpass, text="Plot both teams Missed passes", command=missedPlot).grid(row=1, column=0)
	ttk.Button(missedpass, text="Plot " + df['team_name'][0] + " Missed passes", command=missedPlot_h).grid(row=2, column=0)
	ttk.Button(missedpass, text="Plot " + df['team_name'][1] + " Missed passes", command=missedPlot_a).grid(row=3, column=0)

# Missed passes plot of both teams
def missedPlot():
	(fig,ax) = createPitch(120,80,'yards','gray')
	for i,miss in missed.iterrows():
		x=miss['location'][0]
		y=miss['location'][1]

		team_name=miss['team_name']
		circleSize=2

		if (team_name==df['team_name'][0]):
			missCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="red")     
			missCircle.set_alpha(.2)
		elif (team_name==df['team_name'][1]):
			missCircle=plt.Circle((pitchLengthX-x,y),circleSize,color="blue")      
			missCircle.set_alpha(.2)
		ax.add_patch(missCircle)

	fig.set_size_inches(10, 7)
	plt.show() 

# Missed passes plot of home team
def missedPlot_h():
	(fig,ax) = createPitch(120,80,'yards','gray')
	for i,miss in missed.iterrows():
		x=miss['location'][0]
		y=miss['location'][1]

		team_name=miss['team_name']
		circleSize=2

		if (team_name==df['team_name'][0]):
			missCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="red")     
			missCircle.set_alpha(.2)
			ax.add_patch(missCircle)

	fig.set_size_inches(10, 7)
	plt.show() 

# Missed passes plot of away team
def missedPlot_a():
	(fig,ax) = createPitch(120,80,'yards','gray')
	for i,miss in missed.iterrows():
		x=miss['location'][0]
		y=miss['location'][1]

		team_name=miss['team_name']
		circleSize=2

		if (team_name==df['team_name'][1]):
			missCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="blue")     
			missCircle.set_alpha(.2)
			ax.add_patch(missCircle)

	fig.set_size_inches(10, 7)
	plt.show()

#####--- Additional Statistics ---#####
# Shots analysis in a match
def inDepth():
	depth = Toplevel()
	global shots_df_h
	global shots_df_a
	global Home_S
	global goals_H
	global Home_G 
	global Away_S
	global goals_A
	global Away_G

	shots = df.loc[df['type_name'] == 'Shot'].set_index('id')
	shots_df = pd.DataFrame(columns=["team_name","X","Y","Location","Angle","Distance","Goal"])
	# Split the location column to X and Y coordinates
	for i,shot in shots.iterrows():
		shots_df.at[i,"X"] = shot["location"][0]
		shots_df.at[i,"Y"] = shot["location"][1]
		shots_df.at[i,"team_name"] = shot["team_name"]

	# Create a Goal column, 1 == GOAL, 0 == NOT A GOAL
	for i,shot in shots.iterrows():
		if shot["shot_outcome_name"] == "Goal":
			shots_df.at[i,"Goal"] = 1
		else:
			shots_df.at[i,"Goal"] = 0

	# Create Distance and Angle columns (Explained in the project report)
	for i,shot in shots_df.iterrows():

		shots_df.at[i,'X'] = 120-shot["X"]
		x = shots_df.at[i,'X'] 
		y = abs(shots_df.at[i,'Y'] - 40)

		shots_df.at[i,'Distance']=np.sqrt(x**2 + y**2)
		
		a = np.arctan(8 * x /(x**2 + y**2 - (8/2)**2))
		if a<0:
			a=np.pi+a
		shots_df.at[i,'Angle'] = a

	shots_df_h = shots_df[shots_df["team_name"] == df["team_name"][0]]
	shots_df_a = shots_df[shots_df["team_name"] == df["team_name"][1]]
	#Two dimensional histogram
	Home_S = np.histogram2d(shots_df_h['X'], shots_df_h['Y'],bins=50,range=[[0, 120],[0, 80]])
	goals_H = shots_df_h[shots_df_h['Goal']==1]
	Home_G = np.histogram2d(goals_H['X'], goals_H['Y'],bins=50,range=[[0, 120],[0, 80]])

	Away_S = np.histogram2d(shots_df_a['X'], shots_df_a['Y'],bins=50,range=[[0, 120],[0, 80]])
	goals_A = shots_df_a[shots_df_a['Goal']==1]
	Away_G = np.histogram2d(goals_A['X'], goals_A['Y'],bins=50,range=[[0, 120],[0, 80]])

	firstTeam = ttk.Label(depth, text=home_team + " :")
	firstTeam.grid(row=0, column=0, padx=5)

	GvL = ttk.Button(depth, text="Goals vs Distance", command=SvD_H)
	GvL.grid(row=1, column=0, padx=5)

	GvAA = ttk.Button(depth, text="Goals vs Angle", command=AngleH)
	GvAA.grid(row=1, column=1, padx=5)

	ttk.Separator(depth,orient=HORIZONTAL).grid(row=3, columnspan=2, sticky="ew", pady=5)

	secondTeam = ttk.Label(depth, text=away_team + " :")
	secondTeam.grid(row=4, column=0, padx=5)

	GvL2 = ttk.Button(depth, text="Goals vs Distance", command=SvD_A)
	GvL2.grid(row=5, column=0, padx=5)

	GvAA2 = ttk.Button(depth, text="Goals vs Angle", command=AngleA)
	GvAA2.grid(row=5, column=1, padx=5)


def AngleH():
	fig,ax=plt.subplots(num=1)
	ax.plot(shots_df_h['Angle']*180/np.pi, shots_df_h['Goal'], linestyle='none', marker= '.', markersize= 12, color='black')
	ax.set_ylabel('Goal scored')
	ax.set_xlabel("Shot angle (degrees)")
	plt.ylim((-0.05,1.05))
	ax.set_yticks([0,1])
	ax.set_yticklabels(['No','Yes'])
	plt.show()

def AngleA():
	fig,ax=plt.subplots(num=1)
	ax.plot(shots_df_a['Angle']*180/np.pi, shots_df_a['Goal'], linestyle='none', marker= '.', markersize= 12, color='black')
	ax.set_ylabel('Goal scored')
	ax.set_xlabel("Shot angle (degrees)")
	plt.ylim((-0.05,1.05))
	ax.set_yticks([0,1])
	ax.set_yticklabels(['No','Yes'])
	plt.show()

def SvD_A():
	scatplt, ax = plt.subplots()
	ax = sns.scatterplot(x="Distance", y="Goal",data=shots_df_a)
	scatplt.set_size_inches(10, 7)
	plt.show()

def SvD_H():
	scatplt, ax = plt.subplots()
	ax = sns.scatterplot(x="Distance", y="Goal",data=shots_df_h)
	scatplt.set_size_inches(10, 7)
	plt.show()

# Number of corners
def corners():
	d = df.loc[ df["play_pattern_name"]=='From Corner']
	homeCorners = awayCorners = 0
	for i,corner in d.iterrows():
	    x=corner['location'][0]
	    y=corner['location'][1]
	    if x == 120 and corner["team_name"] == df["team_name"][0]:
	        homeCorners += 1
	    elif x == 120 and corner["team_name"] == df["team_name"][1]:
	        awayCorners += 1
	corners = {df["team_name"][0]:homeCorners, df["team_name"][1]:awayCorners}
	plt.bar(*zip(*corners.items()), color=['#348ABD','#E24A33'])
	plt.show()

# Number of fouls committed 
def fouldcommitted():
	fouls = df.loc[df['type_name'] == 'Foul Committed'].set_index("id")
	countplt, ax = plt.subplots(figsize = (8,5))
	ax = sns.countplot(y = fouls.possession_team_name)
	ax.set_title('Fouls Commited',fontsize = 18, fontweight='bold' )
	ax.set_xlabel('Number of fouls', fontsize = 15)
	ax.set_ylabel('Teams', fontsize = 15)
	for rect in ax.patches:
	    ax.text(rect.get_width(), rect.get_height() + rect.get_y() - 0.4,  rect.get_width())
	plt.show()

#Number of passes
def passe():
	passe = df.loc[df['type_name'] == 'Pass'].set_index("id")
	countplt, ax = plt.subplots(figsize = (8,5))
	ax = sns.countplot(y = passe.team_name)
	ax.set_title('Passes',fontsize = 18, fontweight='bold' )
	ax.set_xlabel('Number of passes', fontsize = 15)
	ax.set_ylabel('Teams', fontsize = 15)
	for rect in ax.patches:
	    ax.text(rect.get_width(), rect.get_height() + rect.get_y() - 0.4,  rect.get_width())
	plt.show()

# Number of goal attempts
def goalattempts():
	attempt = df.loc[df['type_name'] == 'Shot'].set_index('id')

	countplt, ax = plt.subplots(figsize = (8,5))
	ax = sns.countplot(y = attempt.team_name)
	ax.set_title('Attempts',fontsize = 18, fontweight='bold' )
	ax.set_xlabel('Number of attempts', fontsize = 15)
	ax.set_ylabel('Teams', fontsize = 15)
	for rect in ax.patches:
	    ax.text(rect.get_width(), rect.get_height() + rect.get_y() - 0.4,  rect.get_width())
	plt.show()

# Shots heatmap
def shotsHeatMap():
	heatmap_choice = Toplevel(matchanalysis)
	heatmap_choice.title('Choose a team')
	heatmap_choice.geometry('305x50')
	heatmap_choice.iconbitmap('favicon.ico')

	ttk.Button(heatmap_choice, text=df['team_name'][0], command=h_heatmap).grid(row=0, column=0, padx=5, pady=5)
	ttk.Button(heatmap_choice, text=df['team_name'][1], command=a_heatmap).grid(row=0, column=1, padx=5, pady=5)
 
def h_heatmap():
	shots = df.loc[df['type_name'] == 'Shot'].set_index('id')
	(fig,ax) = createPitch(pitchLengthX,pitchWidthY,'yards','gray')

	x=[]
	y=[]
	for i,shot in shots.iterrows():
		if shot['team_name'] == df['team_name'][0]:
			x.append(shot['location'][0])
			y.append(pitchWidthY-shot['location'][1])

	H_Shot=np.histogram2d(y, x,bins=5,range=[[0, pitchWidthY],[0, pitchLengthX]])

	pos=ax.imshow(H_Shot[0], extent=[0,120,0,80], aspect='auto',cmap=plt.cm.Reds)
	fig.colorbar(pos, ax=ax)
	plt.xlim((-1,121))
	plt.ylim((83,-3))
	plt.tight_layout()
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()

def a_heatmap():
	shots = df.loc[df['type_name'] == 'Shot'].set_index('id')
	(fig,ax) = createPitch(pitchLengthX,pitchWidthY,'yards','gray')

	x=[]
	y=[]
	for i,shot in shots.iterrows():
		if shot['team_name'] == df['team_name'][1]:
			x.append(shot['location'][0])
			y.append(pitchWidthY-shot['location'][1])

	H_Shot=np.histogram2d(y, x,bins=5,range=[[0, pitchWidthY],[0, pitchLengthX]])

	pos=ax.imshow(H_Shot[0], extent=[0,120,0,80], aspect='auto',cmap=plt.cm.Reds)
	fig.colorbar(pos, ax=ax)
	plt.xlim((-1,121))
	plt.ylim((83,-3))
	plt.tight_layout()
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()

#####--- Player Statistics ---#####
def overall():
	overall_window = Toplevel(matchanalysis)
	overall_window.title('Overall Stats of ' + player.get())
	overall_window.geometry('400x250')
	overall_window.iconbitmap('favicon.ico')
	
	p = df.loc[df["player_name"] == player.get() ]
	x = p["type_name"].value_counts()
	ttk.Label(overall_window, text = player.get(), font=('Helvatica',12,'bold')).pack()
	subtitle = "Plays as a " + p['position_name'].unique()+ " for " + p['team_name'].unique()
	ttk.Label(overall_window, text= subtitle[0]).pack()
	try:
		if('Pass' in x):
			ttk.Label(overall_window, text="Passes made: " + str(x['Pass'])).pack()
		if('Shot' in x):
			ttk.Label(overall_window, text="Shots made: " + str(x['Shot'])).pack()
		if('Ball Receipt*' in x):
			ttk.Label(overall_window, text="Ball Receipt: " + str(x['Ball Receipt*'])).pack()
		if('Miscontrol' in x):
			ttk.Label(overall_window, text="Miscontrol: " + str(x['Miscontrol'])).pack()
		if('Foul Committed' in x):
			ttk.Label(overall_window, text="Fouls Committed: " + str(x['Foul Committed'])).pack()
		if('Ball Recovery' in x):
			ttk.Label(overall_window, text="Ball Recovery: " + str(x['Ball Recovery'])).pack()
		if('Clearance' in x):	
			ttk.Label(overall_window, text="Clearance: "  + str(x['Clearance'])).pack()
		if('Dribble' in x):
			ttk.Label(overall_window, text="Dribbles made: " + str(x['Dribble'])).pack()
	except Exception as e:
		ttk.Label(overall_window, text = "No data available due to one of these reasons: \n*" + player.get() + " doesn't have special statistics in this match.\n*" +  player.get() + " didn't play in this match. \n*You entered the wrong name(check lineup for correct name)").pack()
		
# Locations of passes
def passes():
	passes = df.loc[df['type_name'] == 'Pass'].set_index('id')
	(fig,ax) = createPitch(120,80,'yards','gray')
	for i, thepass in passes.iterrows():
	    x = thepass['location'][0]
	    y = thepass['location'][1]
	    dx=thepass['pass_end_location'][0]
	    dy=thepass['pass_end_location'][1]
	    missed = (thepass['pass_outcome_name'] == 'Incomplete') | (thepass['pass_outcome_name'] == 'Out') | (thepass['pass_outcome_name'] == 'Unknown')
	    if thepass['player_name'] == player.get() and thepass['team_name'] == home_team:
	        if missed:
	            passCircle = plt.Circle((x,y),2,color="blue")
	            passCircle.set_alpha(.2)
	            ax.add_patch(passCircle)
	            passArrow=plt.Arrow(x,y,dx-x,dy-y,width=3, color="blue")
	            passArrow.set_alpha(.2)
	            ax.add_patch(passArrow)
	        else:
	            passCircle = plt.Circle((x,y),2,color="blue")
	            passCircle.set_alpha(.2)
	            ax.add_patch(passCircle)
	            passArrow=plt.Arrow(x,y,dx-x,dy-y,width=3, color="blue")
	            ax.add_patch(passArrow)
	    elif thepass['player_name'] == player.get() and thepass['team_name'] == away_team:
	        if missed:
	            passCircle = plt.Circle((x,y),2,color="red")
	            passCircle.set_alpha(.2)
	            ax.add_patch(passCircle)

	            passArrow=plt.Arrow(x,y,dx-x,dy-y,width=3, color="red")
	            passArrow.set_alpha(.2)
	            ax.add_patch(passArrow)
	        else:
	            passCircle = plt.Circle((x,y),2,color="red")
	            passCircle.set_alpha(.2)
	            ax.add_patch(passCircle)
	            passArrow=plt.Arrow(x,y,dx-x,dy-y,width=3, color="red")
	            ax.add_patch(passArrow)
	fig.set_size_inches(10,7)
	plt.show()

# Shots locations
def shotsLocations():
	player_shot = df.loc[df['type_name'] == 'Shot'].set_index('id')
	passes = df.loc[df['type_name'] == 'Pass']
	(fig,ax) = createPitch(120,80,'yards','gray')
	circleSize = 2
	for i, theshot in player_shot.iterrows():
		x=theshot['location'][0]
		y=theshot['location'][1]
		goal=theshot['shot_outcome_name']=='Goal'
		i = theshot['pass_assisted_shot_id']
		if theshot['player_name'] == player.get() and theshot['team_name'] == home_team:
			if goal:
				shotCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="blue")
				plt.text((x+1),pitchWidthY-y+1,theshot['player_name']) 
				plt.text((x+1),pitchWidthY-y-6,str(theshot['minute'])+':'+str(theshot['second']), size='small', color='gray')
				if theshot['pass_goal_assist'] == True:
					b = passes.loc[passes['id']== i].set_index('id')
					plt.text((x+1),pitchWidthY-y-3, 'Assisted by: ' + b['player_name'][0], size='small', color='gray')
				ax.add_patch(shotCircle)
			else:
				shotCircle=plt.Circle((x,pitchWidthY-y),circleSize,color="blue")     
				shotCircle.set_alpha(.2)
				ax.add_patch(shotCircle)
		elif theshot['player_name'] == player.get() and theshot['team_name'] == away_team:
			if goal:
				shotCircle=plt.Circle((pitchLengthX-x,y),circleSize,color="red")  
				plt.text((120-x+1), y+1, theshot['player_name'])
				plt.text((120-x+1),y-6,str(theshot['minute'])+':'+str(theshot['second']), size='small', color='gray')
				if theshot['pass_goal_assist'] == True:
					b = passes.loc[passes['id']== i].set_index('id')
					plt.text((120-x+1),y-3, 'Assisted by: ' + b['player_name'][0], size='small', color='gray')    
				ax.add_patch(shotCircle)
			else:
				shotCircle=plt.Circle((pitchLengthX-x,y),circleSize,color="red")      
				shotCircle.set_alpha(.2)
				ax.add_patch(shotCircle)
	fig.set_size_inches(10, 7)
	plt.show()

# Player heatmap
def playerHeatmap():
	passes = df.loc[df['type_name'] == 'Pass']
	shots = df.loc[df['type_name'] == 'Shot']
	receipt = df.loc[df['type_name'] == 'Ball Receipt*']
	duel = df.loc[df['type_name'] == 'Duel']
	dribble = df.loc[df['type_name'] == 'Dribble']
	foul = df.loc[df['type_name'] == 'Foul Committed']
	recover = df.loc[df['type_name'] == 'Ball Recovery']
	clear = df.loc[df['type_name'] == 'Clearance']
	carry = df.loc[df['type_name'] == 'Carry']

	x=[]
	y=[]
	for i,apass in passes.iterrows():
	    if apass['player_name'] == player.get() and 'Pass' in df['type_name'].unique():
	        x.append(apass['location'][0])
	        y.append(pitchWidthY-apass['location'][1])
	        
	for i,ashot in shots.iterrows():
	    if ashot['player_name'] == player.get() and 'Shot' in df['type_name'].unique():
	        x.append(ashot['location'][0])
	        y.append(pitchWidthY-ashot['location'][1])
	        
	for i,areceipt in receipt.iterrows():
	    if areceipt['player_name'] == player.get() and 'Ball Receipt*' in df['type_name'].unique():
	        x.append(areceipt['location'][0])
	        y.append(pitchWidthY-areceipt['location'][1])
	        
	for i,aduel in duel.iterrows():
	    if aduel['player_name'] == player.get() and 'Duel' in df['type_name'].unique():
	        x.append(aduel['location'][0])
	        y.append(pitchWidthY-aduel['location'][1])
	        
	for i,adri in dribble.iterrows():
	    if adri['player_name'] == player.get() and 'Dribble' in df['type_name'].unique():
	        x.append(adri['location'][0])
	        y.append(pitchWidthY-adri['location'][1])
	        
	for i,afoul in foul.iterrows():
	    if afoul['player_name'] == player.get() and 'Shot' in df['type_name'].unique():
	        x.append(afoul['location'][0])
	        y.append(pitchWidthY-afoul['location'][1])
	        
	for i,areco in recover.iterrows():
	    if areco['player_name'] == player.get() and 'Ball Receipt*' in df['type_name'].unique():
	        x.append(areco['location'][0])
	        y.append(pitchWidthY-areco['location'][1])
	        
	for i,aclear in clear.iterrows():
	    if aclear['player_name'] == player.get() and 'Duel' in df['type_name'].unique():
	        x.append(aclear['location'][0])
	        y.append(pitchWidthY-aclear['location'][1])
	                
	for i,acarry in carry.iterrows():
	    if acarry['player_name'] == player.get() and 'Duel' in df['type_name'].unique():
	        x.append(acarry['location'][0])
	        y.append(pitchWidthY-acarry['location'][1])
	        
	        
	#Make a histogram of passes
	H_Pass = np.histogram2d(y, x,bins=5,range=[[0, pitchWidthY],[0, pitchLengthX]])

	(fig,ax) = createPitch(pitchLengthX,pitchWidthY,'yards','gray')
	pos=ax.imshow(H_Pass[0], extent=[0,120,0,80], aspect='auto',cmap=plt.cm.Reds)
	fig.colorbar(pos, ax=ax)
	plt.xlim((-1,121))
	plt.ylim((83,-3))
	plt.tight_layout()
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()

# Dribbles locations
def dribblesLocations():
	dribbles = df.loc[df['type_name']=="Dribble"].set_index("id")
	(fig,ax) = createPitch(120,80,'yards','gray')
	for i, dribble in dribbles.iterrows():
		x=dribble['location'][0]
		y=dribble['location'][1]
		success = dribble["dribble_outcome_name"] == "Complete"
		if dribble['player_name'] == player.get() and dribble['team_name'] == df['team_name'][0]:
			if success:
				dCircle=plt.Circle((x,pitchWidthY-y),2,color="red")     
				ax.add_patch(dCircle)
			else:
				dCircle=plt.Circle((x,pitchWidthY-y),2,color="red") 
				dCircle.set_alpha(.2)
				ax.add_patch(dCircle)
		elif dribble['player_name'] == player.get() and dribble['team_name'] == df['team_name'][1]:
			if success:
				dCircle=plt.Circle((x,pitchWidthY-y),2,color="blue")     
				ax.add_patch(dCircle)
			else:
				dCircle=plt.Circle((x,pitchWidthY-y),2,color="blue") 
				dCircle.set_alpha(.2)
				ax.add_patch(dCircle)
	fig.set_size_inches(10,7)
	plt.show()

#####--- Menu bar ---#####

def aboutus():
	global img
	aboutus = Toplevel(root)
	aboutus.title("About us")
	aboutus.geometry("500x500")
	aboutus.iconbitmap('favicon.ico')

	title0 = Label(aboutus, text="End of Studies Project - Projet Fin d'études", font=("Helvatica",16))
	title2 = Label(aboutus, text="Master Data Science and Intelligent Systems", font=("Helvatica",14))

	canvas = Canvas(aboutus, width = 256, height = 250)  
	canvas.pack()  
	img = ImageTk.PhotoImage(Image.open("logo.png"))  
	canvas.create_image(20, 20, anchor=NW, image=img) 

	title0.pack()
	title2.pack()

def guide():
	guide = Toplevel(root)
	guide.title('Guide')
	guide.geometry('400x100')
	guide.iconbitmap('favicon.ico')
	ttk.Label(guide, text="This feature will be added in the second version.").pack()

def contacts():
	contact = Toplevel(root)
	contact.title('Contacts')
	contact.geometry('400x400')
	contact.iconbitmap('favicon.ico')
	ttk.Label(contact, text="Supervisor: Prof. Esbai Redouane", font=("Helvatica", 12,"bold")).pack()
	ttk.Label(contact, text="Omar El Yousfi: ", font=("Helvatica", 12,"bold")).pack()
	ttk.Label(contact, text="Personal E-mail: elyousfiomarr@gmail.com").pack()
	ttk.Label(contact, text="Academic E-mail: omar.elyousfi@ump.ac.ma").pack()
	ttk.Label(contact, text="Phone number: +212668739696").pack()
	ttk.Label(contact, text="Faculty administration phone number: 05363-58981").pack()

#####--- Possession ---#####
def possession():
	figure = plt.Figure(figsize=(6,5), dpi=100)
	possession = pd.DataFrame(df['possession_team_name'].value_counts())
	chart_type = FigureCanvasTkAgg(figure, root)
	chart_type.get_tk_widget().grid(row=10, column=0)
	possession.plot.pie(subplots=True, figsize=(16,8))
	plt.show()

### ------ Interface --------- ###

def gameLookup():
	global df
	global home_team
	global away_team
	global player
	global competitions
	global matches
	global team_lookup
	global var
	global tkvar
	global lookup
	global matchanalysis

	# Theme
	if tkvar.get() == "Light Theme":
		style = Style(theme="sandstone", themes_file = "ttkbootstrap_themes.json")
	elif tkvar.get() == "Dark Theme":
		style = Style(theme="darktheme", themes_file = "ttkbootstrap_themes.json")

	window = style.master

	#Create new window
	matchanalysis = Toplevel(root)
	matchanalysis.title('Football Analytics v1.')
	matchanalysis.geometry("970x780+300+30")

	# Fixed window size
	matchanalysis.minsize(970, 780)
	matchanalysis.maxsize(970, 780)

	# Favicon
	matchanalysis.iconbitmap('favicon.ico')


	# Import Data
	av = []
	available = os.listdir("events")
	for a in available:
		av.append(a.strip(".json"))
	
	if lookup.get() in av:
		file_name = "events\\" + str(lookup.get()) + '.json'
	else:
		messagebox.showinfo("Not available", "Wrong ID, check ID/Match file below")
		matchanalysis.destroy()

	with open(file_name, encoding='utf-8') as data_file:
		data = json.load(data_file)
	df = pd.json_normalize(data, sep = "_").assign(match_id = file_name[:-5])
	
	match_id = int(lookup.get())
	competitions_id = [2,11,16,37,43,49,72]

	for competi in competitions_id:
		files = os.listdir('matches\\' + str(competi) )
		for file in files:
			with open('matches\\'+ str(competi) + "\\" + file, encoding='utf-8') as f:
				temp = json.load(f)
				for i in range(len(temp)):
					mmid = temp[i]["match_id"]
					if  mmid == match_id:
						matches = temp[i]

	# Home & Away Teams
	home_team = df["team_name"][0]
	away_team = df["team_name"][1]

	# Game Analysis
	params = ttk.LabelFrame(matchanalysis, text="What do you want to analyze?")
	params.grid(row=1, column=0)

	shots_btn = ttk.Button(params, text="Shots", command=shots_locations)
	shots_btn.grid(row=0, column=0, padx=40, pady=5, ipadx=44)

	tackles_btn = ttk.Button(params, text="Cards", command=tackles)
	tackles_btn.grid(row=3, column=0, padx=40, pady=5, ipadx=44)

	fouls_btn = ttk.Button(params, text="Fouls", command=fouls)
	fouls_btn.grid(row=4, column=0, padx=40, pady=5, ipadx=45)

	missedPassesPlot_btn = ttk.Button(params, text="Missed passes", command=missedPassesPlot)
	missedPassesPlot_btn.grid(row=6, column=0, padx=40, pady=5, ipadx=17)

	dribbles = ttk.Button(params, text="Dribbles", command=analyzeDribbles)
	dribbles.grid(row=5, column=0, padx=40, pady=5, ipadx=37)

	possession_analysis = ttk.Button(params, text="Possession Analysis", command=possAnalysis)
	possession_analysis.grid(row=7, column=0, padx=40, pady=5)

	# Additional Statistics
	stats = ttk.LabelFrame(matchanalysis, text="Additional Statistics")
	stats.grid(row=2, column=0, columnspan=3)

	corners_btn = ttk.Button(stats, text="Corners", command=corners)
	corners_btn.grid(row=0, column=0, padx=10, pady=5)

	foulscommitted_btn = ttk.Button(stats, text="Fouls Committed", command=fouldcommitted)
	foulscommitted_btn.grid(row=0, column=1, padx=5, pady=5)

	passes_btn = ttk.Button(stats, text="Passes", command=passe)
	passes_btn.grid(row=0, column=2, padx=5, pady=5)

	goalattempts_btn = ttk.Button(stats, text="Goal Attempts", command=goalattempts)
	goalattempts_btn.grid(row=0, column=3, padx=5, pady=5)

	shots_heatmap = ttk.Button(stats, text="Shots heatmap", command=shotsHeatMap)
	shots_heatmap.grid(row=0, column=4, padx=5, pady=5)

	shotsVsDis = ttk.Button(stats, text="Shots In-Depth", command=inDepth)
	shotsVsDis.grid(row=0, column=5, padx=5, pady=5)

	# Player Statistics
	playerStats = ttk.LabelFrame(params, text="Player Statistics")
	playerStats.grid(row=8, column=0, pady=2, padx=2, columnspan=4)

	player = ttk.Entry(playerStats)
	player.grid(row=0, column=0, columnspan=4)

	overall_btn = ttk.Button(playerStats, text="Overall Stats", command=overall)
	overall_btn.grid(row=1, column=0, padx=10, pady=5)

	passes_btn = ttk.Button(playerStats, text="Passes", command=passes)
	passes_btn.grid(row=2, column=0, padx=10, pady=5, ipadx=14)

	player_heatmap = ttk.Button(playerStats, text="Heatmap", command=playerHeatmap)
	player_heatmap.grid(row=2, column=1, padx=10, pady=5, ipadx=4)

	s_l = ttk.Button(playerStats, text="Shots Locations", command=shotsLocations)
	s_l.grid(row=3, column=0, padx=10, pady=5, columnspan=4)

	d_l = ttk.Button(playerStats, text="Dribbles", command=dribblesLocations)
	d_l.grid(row=1, column=1, padx=10, pady=5, ipadx=7)


	# Header Info
	teams = ttk.LabelFrame(matchanalysis, text='The game is between:')
	teams.grid(row=0, column=1, pady=5, padx=5)

	homeAway = ttk.Label(teams, font=('Helvatica',12, 'bold'), text="Home Team: " +  df['possession_team_name'].unique()[0] + "      " + str(matches['home_score']) + "\n" + "Away Team: " +df['possession_team_name'].unique()[1] + "      " + str(matches['away_score']))
	homeAway.grid(row=0, column=0, padx=25, pady=15, columnspan=3)
	if('referee' in matches):
		referee = ttk.Label(teams,text="Referee: " + matches['referee']['name'], font=('Helvatica',10,'bold'))
		referee.grid(row = 1, column = 0, padx=25, pady=5, columnspan=3)

	if('competition' in matches):
		competition = ttk.Label(teams, text = "Competition: " + matches["competition"]["competition_name"], font=("Helvatica", 10, 'bold'))
		competition.grid(row=2, column = 0, padx=25, pady=5, columnspan=3)

	if('stadium' in matches):
		stadium = ttk.Label(teams, text = "Stadium: " + matches['stadium']['name'], font=("Helvatica", 10, "bold"))
		stadium.grid(row=4, column=0, padx=25, pady=5, columnspan=3)

	match_date = ttk.Label(teams, text="Date: " + matches['match_date'], font=("Helvatica", 10, "bold"))
	match_date.grid(row=3, column=0, padx=25, columnspan=3)


	pass_btn = ttk.Button(teams, text="Passes Ranking", command=showMostPasses)
	pass_btn.grid(row=5, column=0, pady=5, padx=5)

	teams_btn = ttk.Button(teams, text="Lineup", command=showTeams)
	teams_btn.grid(row=5, column=1, padx=5, pady=5, ipadx=5)

	shot_btn = ttk.Button(teams, text="Shots Ranking", command=shots)
	shot_btn.grid(row=5, column=2, pady=5, padx=5)

	result = ttk.Button(teams, text="Summary", command=summary)
	result.grid(row=6, column=0, pady=5, columnspan=3)

	# Possession
	poss = ttk.LabelFrame(matchanalysis, text='Possession: ')
	poss.grid(row=1, column=1, padx=5)

	figure = plt.Figure(figsize=(4,4), dpi=100)
	ax = figure.add_subplot(111)
	possession = pd.DataFrame(df['possession_team_name'].value_counts())
	chart_type = FigureCanvasTkAgg(figure, poss)
	chart_type.get_tk_widget().grid(row=0, column=0)
	possession.plot(kind='pie', subplots=True, legend=True, ax=ax, autopct='%1.1f%%')


	topPasses = ttk.LabelFrame(matchanalysis, text="TOP5")
	topPasses.grid(row=1, column=2, padx=5)



	pa = ttk.LabelFrame(topPasses, text="Passes Ranking")
	pa.grid(row=0, column=0)

	def Table(r, label):
		lst = list(r.items())
		total_rows = len(lst)
		total_columns = len(lst[0])
		for i in range(total_rows):
			for j in range(total_columns):
				e = Entry(label, width=20, font=('Arial',8,'bold'))
				e.grid(row=i, column=j)
				e.configure(state='normal')
				e.insert(END, lst[i][j])
				e.configure(state='disabled')

	passesRa = df.loc[df['type_name'] == 'Pass'].set_index('id')
	r = dict(passesRa['player_name'].value_counts()[:5])
	t = Table(r, pa)
	

	ttk.Separator(topPasses,orient=HORIZONTAL).grid(row=1, columnspan=5, sticky="ew", pady=5)

	
	d = ttk.LabelFrame(topPasses, text="Dribblers")
	d.grid(row=2, column=0)
	dribbles = df.loc[df['type_name']=="Dribble"].set_index("id")
	succ = dribbles.loc[dribbles["dribble_outcome_name"] == "Complete"]
	top_dribblers = dict(succ["player_name"].value_counts()[:5])

	t = Table(top_dribblers, d)


	ttk.Separator(topPasses,orient=HORIZONTAL).grid(row=3, columnspan=5, sticky="ew", pady=5)

	topMissed = ttk.LabelFrame(topPasses, text="Missed passes")
	topMissed.grid(row=4, column=0)

	missedPasses = df[df['pass_outcome_name'].isin(["Incomplete","Out","Pass Offside", "Unknown"])]
	top_missing = dict(missedPasses["player_name"].value_counts()[:5])

	t = Table(top_missing, topMissed)

	pre = ttk.Frame(matchanalysis)
	pre.grid(row=0, column=0)

	ttk.Label(pre, text="The current match id: ").grid(row=1,column=0)
	e = ttk.Entry(pre)
	e.insert(END, str(lookup.get()))
	e.configure(state='disabled')
	e.grid(row=2, column=0, pady=5, columnspan=5)

	ttk.Button(pre, text="Close", command=matchanalysis.destroy).grid(row=3, column=0, pady=5)


	ttk.Separator(pre,orient=HORIZONTAL).grid(row=4, columnspan=5, sticky="ew", pady=5)

	teams_list = ttk.Button(pre, text="ID/Match", command = teamslist)
	teams_list.grid(row=5, column=0, columnspan=2, pady=10)


	teamAnalysis = ttk.LabelFrame(matchanalysis, text='Analyse a team: ')
	teamAnalysis.grid(row=0, column=2, padx=5)

	team_lookup = ttk.Entry(teamAnalysis)
	team_lookup.grid(row=0, column=0, pady=5, padx=5)

	seasons =['2011/2012', '2012/2013', 
			'2015/2016', '2005/2006', 
			'2006/2007', '2019/2020', 
			'2016/2017', '2010/2011', 
			'2009/2010', '2008/2009', 
			'2003/2004', '2018', 
			'2019', '2014/2015', 
			'2020/2021', '2018/2019', 
			'2004/2005', '2017/2018', 
			'2007/2008', '2013/2014',
			"All"]

	var = StringVar()
	var.set("2019/2020")

	drop = ttk.OptionMenu(teamAnalysis, var, *seasons)
	drop.grid(row=0, column=1, pady=5, padx=1)

	analyzeshots = ttk.Button(teamAnalysis, text="Analyze shots", command=AnalyzeShots)
	analyzeshots.grid(row=1, column=0, pady=2, ipadx=7)

	analyzeshots = ttk.Button(teamAnalysis, text="Top Goal Scorers", command=goalScorers)
	analyzeshots.grid(row=1, column=1, pady=2, ipadx=5)

	submit_team = ttk.Button(teamAnalysis, text="Results", command=analyzeTeam)
	submit_team.grid(row=2, column=0, pady=5)

	team_info = ttk.Button(teamAnalysis, text="Manager", command=teamInfo)
	team_info.grid(row=2, column=1, pady=5)


	ttk.Separator(teamAnalysis,orient=HORIZONTAL).grid(row=3, columnspan=2, sticky="ew", pady=5)

	compe_info = ttk.Button(teamAnalysis, text="Competitions data", command=compeData)
	compe_info.grid(row=4, column=0,padx=2, pady=5)

	teams_nbr = ttk.Button(teamAnalysis, text="Number of matches", command=nbrMatches)
	teams_nbr.grid(row=4, column=1, padx=2, pady=5)

	menu = Menu(matchanalysis)

	#### File #####
	filemenu = Menu(menu)
	#filemenu.add_command(label="Save plot", command=save)
	filemenu.add_separator()
	filemenu.add_command(label='Exit', command=matchanalysis.destroy)
	menu.add_cascade(label="File", menu=filemenu)

	#### Help #####
	helpmenu = Menu(menu)
	helpmenu.add_command(label="About us", command=aboutus)
	helpmenu.add_command(label="Quick guide", command=guide)
	helpmenu.add_command(label="Contacts", command=contacts)
	menu.add_cascade(label="Help", menu=helpmenu)
	# -------------------------------------------------------------------------#

	matchanalysis.config(menu=menu)
	matchanalysis.mainloop()

# -------------------------------------------------------------------------#
style = Style(theme="sandstone", themes_file = "ttkbootstrap_themes.json")
window = style.master

# 00
header = ttk.LabelFrame(root, text="Instructions")
header.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

ttk.Label(header, text="• Insert a match ID to analyze it, if you don't know the ID of the match you are looking for, check the ID/Match file below.").grid(row=0, column=0)
ttk.Label(header, text="• Insert a team name to analyze.").grid(row=1, column=0)
ttk.Label(header, text="• Contact me for any inconvenience: elyousfiomarr@gmail.com").grid(row=2, column=0)


# 01
choices = ['Light Theme', 'Dark Theme']
tkvar = StringVar()

pre = ttk.LabelFrame(root, text='Match Analysis:')
pre.grid(row=1, column=0)

ttk.Label(pre, text="The match id: ").grid(row=1,column=0)
lookup = ttk.Entry(pre)
lookup.grid(row=2, column=0, pady=5, columnspan=5)


theme_choice =  ttk.OptionMenu(pre, tkvar, *choices)
ttk.Label(pre, text="Choose a theme").grid(row=0, column=0, pady=5)
theme_choice.grid(row=0, column=1, pady=5, padx=5)

submit_lookup = ttk.Button(pre, text="Analyze this match", command = gameLookup)
submit_lookup.grid(row=3, column=0,columnspan=5)

ttk.Separator(pre,orient=HORIZONTAL).grid(row=4, columnspan=5, sticky="ew", pady=5)

teams_list = ttk.Button(pre, text="ID/Match", command = teamslist)
teams_list.grid(row=5, column=0, columnspan=2, pady=10)


# 02
teamAnalysis = ttk.LabelFrame(root, text='Analyse a team: ')
teamAnalysis.grid(row=1, column=1)

team_lookup = ttk.Entry(teamAnalysis)
team_lookup.grid(row=0, column=0, pady=5, padx=5)

seasons =['2011/2012', '2012/2013', 
		'2015/2016', '2005/2006', 
		'2006/2007', '2019/2020', 
		'2016/2017', '2010/2011', 
		'2009/2010', '2008/2009', 
		'2003/2004', '2018', 
		'2019', '2014/2015', 
		'2020/2021', '2018/2019', 
		'2004/2005', '2017/2018', 
		'2007/2008', '2013/2014',
		"All"]

var = StringVar()
var.set("2019/2020")

drop = ttk.OptionMenu(teamAnalysis, var, *seasons)
drop.grid(row=0, column=1, pady=5, padx=1)

analyzeshots = ttk.Button(teamAnalysis, text="Analyze shots", command=AnalyzeShots)
analyzeshots.grid(row=1, column=0, pady=2, ipadx=7)

scor = ttk.Button(teamAnalysis, text="Top Goal Scorers", command=goalScorers)
scor.grid(row=1, column=1, pady=2, ipadx=5)

submit_team = ttk.Button(teamAnalysis, text="Results", command=analyzeTeam)
submit_team.grid(row=2, column=0, pady=5)

team_info = ttk.Button(teamAnalysis, text="Manager", command=teamInfo)
team_info.grid(row=2, column=1, pady=5)


ttk.Separator(teamAnalysis,orient=HORIZONTAL).grid(row=3, columnspan=2, sticky="ew", pady=5)

compe_info = ttk.Button(teamAnalysis, text="Competitions data", command=compeData)
compe_info.grid(row=4, column=0,padx=2, pady=5)

teams_nbr = ttk.Button(teamAnalysis, text="Number of matches", command=nbrMatches)
teams_nbr.grid(row=4, column=1, padx=2, pady=5)



# Menu
menu = Menu(root)

#### File #####
filemenu = Menu(menu)
filemenu.add_command(label='Exit', command=root.destroy)
menu.add_cascade(label="File", menu=filemenu)

#### Help #####
helpmenu = Menu(menu)
helpmenu.add_command(label="About us", command=aboutus)
helpmenu.add_command(label="Quick guide", command=guide)
helpmenu.add_command(label="Contacts", command=contacts)
menu.add_cascade(label="Help", menu=helpmenu)
# -------------------------------------------------------------------------#

root.config(menu=menu)
root.mainloop()

# Graduation Project
# By: Omar El Yousfi
# Supervised by: Prof. Redouane Esbai
# Master Data Science & Intelligent Systems
# Poly-Disciplinary Faculty of Nador
