import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
d=pd.read_csv("../data/data.csv")
d['Month']=pd.factorize(d['Month'])[0]+1
X=d[['Month']]
y=d['Sales']
m=LinearRegression()
m.fit(X,y)
f=np.array([[13],[14],[15]])
p=m.predict(f)
print("Future Predictions:")
i=13
for v in p:
    print("Month",i,":",round(v,2))
    i+=1
a=m.score(X,y)
print("Accuracy:",round(a*100,2),"%")
os.makedirs("../output",exist_ok=True)
plt.scatter(X,y)
plt.plot(X,m.predict(X))
plt.scatter(f,p)
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Prediction")
plt.savefig("../output/prediction_graph.png")
plt.show()