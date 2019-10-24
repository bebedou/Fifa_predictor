import pandas as pd
import re
import requests
from bs4 import BeautifulSoup
import numpy as np
import math
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow import keras
"""
TODO: Importer les datasets en dataframe, et train !
Strategie : 4 defenseurs 4 milieux 2 attaquants avec les meilleures notes + gardien

UI : choisir les 2 clubs dans une liste (pays puis club
Choisir les 11 joueurs de la composition
bingo ! 
"""
def read_db_from_csv(filename, sep):
	f_d = pd.read_csv(filename, sep=sep, encoding = "utf-8")
	return f_d
def replace_plus_values(str):
	ret = str
	try :
		if "+" in ret :
			ret = int(str[:2])
		else :
			if str is "":
				ret = 0
	except TypeError:
		if math.isnan(ret):
			ret = 0
		pass
	return ret
def select_position(df):
	columns = ["LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB"]
	sub_df = df[columns]
	#sub_df = sub_df.apply(replace_nan)
	sub_df = sub_df.replace(r'^\s*$', 0, regex=True)
	for i in columns:
		sub_df[i] = sub_df[i].apply(replace_plus_values)
		sub_df[i] = sub_df[i].astype('int')
		sub_df[i] = pd.to_numeric(sub_df[i])
	sub_df["GK"] = sub_df["CM"].apply(lambda val : max(int(val), 1))
	sub_df["GK"] = sub_df["GK"].astype('int')
	sub_df["GK"] = pd.to_numeric(sub_df["GK"])
	sub_df["Position"] = sub_df.idxmax(axis=1)
	#sub_df.to_excel("test_1.xlsx")
	return sub_df["Position"]

def position_to_idx(val):
	#IDX = 0 if GK, 1 for Defenders, 2 for midfielders and 3 for forwards
	forward_pos = ["LS", "ST", "RS", "LF", "CF", "RF", "RW", "LW"]
	midfield_pos = ["LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LDM", "CDM", "RDM"]
	defender_pos = ["LWB","RWB", "LB", "LCB", "CB", "RCB", "RB"]
	idx = 0
	if val in forward_pos:
		idx = 3
	if val in midfield_pos:
		idx = 2
	if val in defender_pos:
		idx = 1
	return idx
def apply_reduce_position(df):
	fd =pd.DataFrame()
	fd["reduced_position"] = df["Position"].apply(position_to_idx)
	return fd["reduced_position"]
def construct_usable_training_data(df):
	#reduce dataframe to 11x20 players
	#TODO after this function : sort the data to make it used by the neural network
	tmp_df = df
	new_df = pd.DataFrame(columns=["Name", "Club", "Overall", "Position", "reduced_position"])
	curr_index = 0
	
	teams = ["Manchester United", "Leicester City", "Bournemouth", "Cardiff City", "Fulham",
			"Crystal Palace", "Huddersfield Town", "Chelsea", "Newcastle United", "Tottenham Hotspur",
			"Watford", "Brighton & Hove Albion", "Wolverhampton Wanderers", "Everton", "Arsenal", 
			"Manchester City", "Liverpool", "West Ham United", "Southampton", "Burnley"]
	for i in teams: 
		#add GK to DF
		GK_df = tmp_df.loc[tmp_df["Club"].isin([i])].loc[tmp_df["reduced_position"].isin([0])]
		#print (GK_df)
		idx = GK_df["Overall"].idxmax()
		res = GK_df.loc[[idx]]
		new_df = new_df.append(res)
		#Add 4 defenders to dataframe
		def_df = tmp_df.loc[tmp_df["Club"].isin([i])].loc[tmp_df["reduced_position"].isin([1])]
		idx = def_df["Overall"].idxmax()
		res = def_df.loc[[idx]]
		new_df = new_df.append(res)
		#drop defender already added
		def_df = def_df.drop(idx)
		#add def number 2
		idx = def_df["Overall"].idxmax()
		res = def_df.loc[[idx]]
		new_df = new_df.append(res)
		#drop defender already added
		def_df = def_df.drop(idx)
		#add def number 3
		idx = def_df["Overall"].idxmax()
		res = def_df.loc[[idx]]
		new_df = new_df.append(res)
		#drop defender already added
		def_df = def_df.drop(idx)
		#add def number 4
		idx = def_df["Overall"].idxmax()
		res = def_df.loc[[idx]]
		new_df = new_df.append(res)
		#drop defender already added
		def_df = def_df.drop(idx)
		
		
		#Add 3 midfielders to dataframe
		mid_df = tmp_df.loc[tmp_df["Club"].isin([i])].loc[tmp_df["reduced_position"].isin([2])]
		idx = mid_df["Overall"].idxmax()
		res = mid_df.loc[[idx]]
		new_df = new_df.append(res)
		#drop midfielder already added
		mid_df = mid_df.drop(idx)
		#add midfielder number 2
		idx = mid_df["Overall"].idxmax()
		res = mid_df.loc[[idx]]
		new_df = new_df.append(res)
		#drop midfielder already added
		mid_df = mid_df.drop(idx)
		#add midfielder number 3
		idx = mid_df["Overall"].idxmax()
		res = mid_df.loc[[idx]]
		new_df = new_df.append(res)
		#drop midender already added
		mid_df = mid_df.drop(idx)
		
		
		#Add 3 forwards to dataframe
		fw_df = tmp_df.loc[tmp_df["Club"].isin([i])].loc[tmp_df["reduced_position"].isin([3])]
		idx = fw_df["Overall"].idxmax()
		res = fw_df.loc[[idx]]
		new_df = new_df.append(res)
		#drop midfielder already added
		fw_df = fw_df.drop(idx)
		#add forward number 2
		idx = fw_df["Overall"].idxmax()
		res = fw_df.loc[[idx]]
		new_df = new_df.append(res)
		#drop midfielder already added
		fw_df = fw_df.drop(idx)
		#add forward number 3
		idx = fw_df["Overall"].idxmax()
		res = fw_df.loc[[idx]]
		new_df = new_df.append(res)
		#drop midfielder already added
		fw_df = fw_df.drop(idx)
	new_df = new_df.reset_index(drop=True)
	print (new_df)
	return new_df
def get_players_overall_from_df(df, reduced_pos, nb):
	list = []
	tab = df.loc[df["reduced_position"].isin([reduced_pos])]
	tab = tab.to_numpy()
	for i in range (0, nb):
		list.append(tab[i][2])
	return list
def construct_training_data(features, targets):
	#iterate over the rows of the results and construct proper dataframe with players overall and game score
	columns = ["HTGK", "HTDF1", "HTDF2", "HTDF3", "HTDF4", "HTMF1", "HTMF2", "HTMF3", "HTFW1", "HTFW2", "HTFW3", 
	"ATGK", "ATDF1", "ATDF2", "ATDF3", "ATDF4", "ATMF1", "ATMF2", "ATMF3", "ATFW1", "ATFW2", "ATFW3", "HTG", "ATG"]
	training_df= pd.DataFrame(columns=columns)
	
	for index, row in targets.iterrows():
		#print (row["HomeTeam"], row["FTHG"], row["FTAG"] ,row["AwayTeam"])
		row_to_append = []
		#add proper players in the list and append the list to the dataframe
		#get home team players
		hteam_df = features.loc[features["Club"].isin([row["HomeTeam"]])]
		ateam_df = features.loc[features["Club"].isin([row["AwayTeam"]])]
		#get HomeTeam players overall
		row_to_append = row_to_append + get_players_overall_from_df(hteam_df, 0, 1)
		row_to_append = row_to_append + get_players_overall_from_df(hteam_df, 1, 4)
		row_to_append = row_to_append + get_players_overall_from_df(hteam_df, 2, 3)
		row_to_append = row_to_append + get_players_overall_from_df(hteam_df, 3, 3)
		#Get AwayTeam players overall
		row_to_append = row_to_append + get_players_overall_from_df(ateam_df, 0, 1)
		row_to_append = row_to_append + get_players_overall_from_df(ateam_df, 1, 4)
		row_to_append = row_to_append + get_players_overall_from_df(ateam_df, 2, 3)
		row_to_append = row_to_append + get_players_overall_from_df(ateam_df, 3, 3)
		#row_to_append.append(row["HomeTeam"])
		#row_to_append.append(row["AwayTeam"])
		row_to_append.append(row["FTHG"])
		row_to_append.append(row["FTAG"])
		training_df = training_df.append(pd.DataFrame([row_to_append], columns=columns))
		#print (row_to_append)
	print (training_df)
	return training_df
class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch % 100 == 0: print('')
		print('.', end='')
def plot_history(history):
	hist = pd.DataFrame(history.history)
	hist['epoch'] = history.epoch

	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Abs Error [MPG]')
	plt.plot(hist['epoch'], hist['mae'],
		   label='Train Error')
	plt.plot(hist['epoch'], hist['val_mae'],
		   label = 'Val Error')
	plt.ylim([0,5])
	plt.legend()

	plt.figure()
	plt.xlabel('Epoch')
	plt.ylabel('Mean Square Error [$MPG^2$]')
	plt.plot(hist['epoch'], hist['mse'],
		   label='Train Error')
	plt.plot(hist['epoch'], hist['val_mse'],
		   label = 'Val Error')
	plt.ylim([0,20])
	plt.legend()
	plt.show()

def train_model():
	# Load data from csv files, to train we wil use the results and players ratings from BPL 18-19 & FIFA 19
	training_columns = ["HTGK", "HTDF1", "HTDF2", "HTDF3", "HTDF4", "HTMF1", "HTMF2", "HTMF3", "HTFW1", "HTFW2", "HTFW3", 
	"ATGK", "ATDF1", "ATDF2", "ATDF3", "ATDF4", "ATMF1", "ATMF2", "ATMF3", "ATFW1", "ATFW2", "ATFW3"]
	target_columns = [ "HTG", "ATG"]
	#l1_results_df = read_db_from_csv("resultats-ligue-1-18-19.csv", ";")
	players_df = read_db_from_csv("players_data_19.csv", ",")
	results_df = read_db_from_csv("pl_season-1819.csv", ",")
	#results_df = pd.concat([results_df, l1_results_df)]
	results_df = results_df.reset_index(drop=True)
	#print(players_df.describe())
	#print(results_df.describe())
	targets = results_df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
	#print(targets.describe())
	players_df["Position"] = select_position(players_df)
	features = players_df[["Name", "Club", "Overall", "Position"]]
	features["reduced_position"]= apply_reduce_position(features)
	#print (features.loc[features["Club"].isin(["Southampton"])])
	tmp = construct_usable_training_data(features)
	training_data = construct_training_data(tmp, targets)
	training_data = training_data.reindex(np.random.permutation(training_data.index))
	#select training and test data from dataframe
	test_data = training_data.tail(50)
	test_features = test_data[training_columns]
	test_targets = test_data[target_columns]
	training_data = training_data.head(330)
	training_features = training_data[training_columns]
	training_targets = training_data[target_columns]
	
	model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(22, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)])

	optimizer = tf.keras.optimizers.RMSprop(0.002)
	#optimizer = tf.keras.optimizers.Adam()
	#loss = tf.keras.losses.MeanAbsolutePercentageError()
	model.compile(loss='mae', optimizer=optimizer, metrics = ["mae", "mse"])

	
	EPOCHS = 500

	history = model.fit(
		tf.convert_to_tensor(training_features.to_numpy(), np.int32), tf.convert_to_tensor(training_targets.to_numpy(), np.int32),
		epochs=EPOCHS, validation_split = 0.1, verbose=0,batch_size = 10,
		callbacks=[PrintDot()], shuffle=True)
	#plot_history(history)
	
	loss, mae, mse = model.evaluate(tf.convert_to_tensor(test_features.to_numpy(), np.int32), tf.convert_to_tensor(test_targets.to_numpy(), np.int32), verbose=2)
	test_predictions = model.predict(tf.convert_to_tensor(test_features.to_numpy(), np.int32))
	print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
	nb_good_winners = 0
	for i in range (0, len(test_predictions)):
	
		if (((int(test_predictions[i][0]) > int(test_predictions[i][1]) and test_targets.to_numpy()[i][0] > test_targets.to_numpy()[i][1])) or
			((int(test_predictions[i][0]) < int(test_predictions[i][1]) and test_targets.to_numpy()[i][0] < test_targets.to_numpy()[i][1])) or
			((int(test_predictions[i][0]) == int(test_predictions[i][1]) and test_targets.to_numpy()[i][0] == test_targets.to_numpy()[i][1]))):
				nb_good_winners = nb_good_winners + 1
			
		print ("predicted score is {}-{}, real score is  {} - {}, correct guesses : {}".format(
		int(test_predictions[i][0]), int(test_predictions[i][1]),test_targets.to_numpy()[i][0], test_targets.to_numpy()[i][1], nb_good_winners))
		
	return model
	
	
def main():
	model = train_model()
	
	
	
	return 0
	
main()