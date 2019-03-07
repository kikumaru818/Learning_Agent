import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_comparition(x,y1,y2,y3,ylim_lower_bound = 0,ylim_upper_bound = 13,
					 label1 = "Gridworld", label2 = "Cart-Pole-3",label3 = "Cart-Pole-5",
					 output_file_name = "plot2"):
	fig, ax = plt.subplots(figsize=(10,5))
	ax.set_ylabel("Square mean td error")  
	ax.set_xlabel("Step size")
	plt.ylim(ylim_lower_bound, ylim_upper_bound)
	ax.set_xscale("log", nonposx='clip') # log (x)
	plt.tight_layout()
	plt.plot(x,y1,"r+",linestyle="-",label=label1)
	plt.plot(x,y2,"b*",linestyle="-",label=label2)
	plt.plot(x,y3,"g^",linestyle="-",label=label3)

	plt.legend()
	fig.savefig(output_file_name+".pdf")  

def plot_comparition2(x,y1,y2,y3,ylim_lower_bound = 0,ylim_upper_bound = 2,
					 label1 = "Gridworld 100", label2 = "grid-world 500",label3 = "grid-world 1000",
					 output_file_name = "plot3"):
	fig, ax = plt.subplots(figsize=(10,5))
	ax.set_ylabel("Square mean td error")
	ax.set_xlabel("Step size")
	plt.ylim(ylim_lower_bound, ylim_upper_bound)
	ax.set_xscale('symlog', nonposx='clip')
	ax.set_aspect('equal')

	ax.set_xscale("log", nonposx='clip') # log (x)
	#plt.tight_layout()
	
	# print(x)
	# print(y2)
	# x = [0.000099999, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
	#  0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
	#  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1] # change it to x[0] = 0.0000000001
	plt.plot(x,y1,"r+",linestyle="-",label=label1)
	plt.plot(x,y2,"b*",linestyle="-",label=label2)
	plt.plot(x,y3,"g^",linestyle="-",label=label3)

	plt.legend()
	fig.savefig(output_file_name+".pdf")


def plot_trails_by_file_name(output_file,title):
	results = pickle.load(open(output_file,"rb"))
	results = np.asarray(results)
	means = np.mean(results, axis = 0)
	stds = np.std(results, axis = 0)
	fit_up = means + stds
	fit_dw = means - stds
	print(results)
	print(means)
	print(stds)
	x = [i for i in range(len(means))]
	fig, ax = plt.subplots(figsize=(10,5))
	ax.set_ylabel("Return")
	ax.set_xlabel("Episode")
	plt.tight_layout()
	# "b" is for blue, lw=1.2 represents the width of line
	plt.plot(x, means, "r", lw= 0.02, label="Mean")
	ax.fill_between(x, fit_up, fit_dw, alpha=.35, label="1-sigma interval")
	# plt.errorbar(x,means,stds,marker='*')
	plt.title(title)
	fig.savefig(output_file+".pdf")

def plot_trails(results,title,ylim_lower_bound=-2000, ylim_upper_bound=0):
	#results = pickle.load(open(output_file,"rb"))
	title = title.replace(".","")
	results = np.asarray(results)
	means = np.mean(results, axis = 0)
	stds = np.std(results, axis = 0)
	fit_up = means + stds
	fit_dw = means - stds
	#print(results)
	#print(means)
	#print(stds)
	x = [i for i in range(len(means))]
	fig, ax = plt.subplots(figsize=(10,5))
	#plt.ylim(ylim_lower_bound, ylim_upper_bound)
	ax.set_ylabel("Return")
	ax.set_xlabel("Episode")
	plt.tight_layout()
	# "b" is for blue, lw=1.2 represents the width of line
	plt.plot(x, means, "r", lw= 0.02, label="Mean")
	ax.fill_between(x, fit_up, fit_dw, alpha=.35, label="1-sigma interval")
	# plt.errorbar(x,means,stds,marker='*')
	plt.title(title)
	fig.savefig(title+".pdf")

#plot_comparition(x[:-1],y1[:-1],y2[:-1],y3[:-1],ylim_upper_bound=15, output_file_name="plot10")

