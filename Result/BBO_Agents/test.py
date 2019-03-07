import pickle
import matplotlib.pyplot as plt
import numpy as np

result = pickle.load(open("q3_trails_result.pkl","rb"))
#theta, _ = pickle.load(open("q2_trails_theta.pkl","rb"))
haha, _,theta = pickle.load(open("q3_trails_theta.pkl","rb"))

list_length = [len(i) for i in result]
list_max = max(list_length)
list_min = min(list_length)
#new_result = [result[i] + [theta]*(list_max-len(result[i])) for i in range(len(result))]
new_result = [result[i][0:list_min] for i in range(len(result))]




result=np.mean(new_result,axis=0)
x = [i for i in range(len(result))]
#print("J_max is : ", result[-1])
plt.plot(x, result)
plt.title("grid world fchc")
plt.show()
#print(result)
