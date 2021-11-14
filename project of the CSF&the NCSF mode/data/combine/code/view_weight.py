import matplotlib.pyplot as plt
import utils as U

features,pros=U.load_weight()

x=range(0,len(features))
plt.bar(x,pros)
plt.xticks(x, features,rotation=30, fontsize='small')
plt.ylabel("The weight")
plt.title("The weight of features")
plt.show()
plt.close()