import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []
def get_data(filename):
    i = 0
    with open(filename, 'r') as csvfile:
        csvfileReader = csv.reader(csvfile, delimiter=';')
        next(csvfileReader)        
        for row in csvfileReader:
            dates.append(i)#int(row[0].split('-')[0]))
            prices.append(float(row[1]))
            i = i + 1
    return i

def predict_prices(dates, prices, x, ):
    dates = np.reshape(dates, (len(dates), 1))
    
    print('starting pricing')
    svr_lin = SVR(kernel='linear', C=1e3)
    #svr_poly = SVR(kernel='poly', C=1e3, degree = 2) # uncomment if you want poly-regression
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin.fit(dates, prices)
    print('lin completed')
    #svr_poly.fit(dates, prices)
    print('poly completed')
    svr_rbf.fit(dates, prices)
    print('svr completed')
    
    futureDates = [x + i for i in range(10)]
    futureDates = np.reshape(futureDates, (len(futureDates), 1))
    
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    #plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polinomial model')
    
    plt.plot(futureDates, svr_rbf.predict(futureDates), color='orange', label='RBF predition model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0] #, svr_poly.predict(x)[0]

i = get_data('UsdRubLight.csv')
predicted_price = predict_prices(dates, prices, i)
print(i)
print(predicted_price)