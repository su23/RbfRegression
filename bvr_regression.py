import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def get_data(filename):
    i = 0
    dates = []
    prices = []
    with open(filename, 'r') as csvfile:
        csvfileReader = csv.reader(csvfile, delimiter=';')
        next(csvfileReader)        
        for row in csvfileReader:
            dates.append(i)
            prices.append(float(row[1]))
            i = i + 1
    return dates, prices, i

def predict_prices(dates, prices, x, fdates, fprices):
    dates = np.reshape(dates, (len(dates), 1))
    
    print('starting pricing')
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(dates, prices)
    print('svr completed')
    
    forecastDates = [i + x for i in range(10)]
    forecastDates = np.reshape(forecastDates, (len(forecastDates), 1))
    fdates = [i +x for i in fdates]
    fdates = np.reshape(fdates, (len(fdates), 1))
    
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    
    plt.plot(forecastDates, svr_rbf.predict(forecastDates), color='pink', label='RBF model')    
    plt.scatter(fdates, fprices, color='orange', label='RBF predition model')
    plt.ylabel('USD\RUB')
    plt.show()
    
   
    plt.scatter(dates[-5:], prices[-5:], color='black', label='Данные')
    plt.plot(dates[-5:], svr_rbf.predict(dates)[-5:], color='red', label='RBF модель')
    plt.plot(forecastDates, svr_rbf.predict(forecastDates + 2), color='red', label='Прогноз')    
    plt.scatter(fdates, fprices, color='orange', label='Реальные данные')
    plt.ylabel('USD\RUB')
    #plt.title('Результаты регрессии')
    plt.legend()
    plt.show()
    
    sum = .0
    M = .0
    for i in range(2):
        M = M + abs(svr_rbf.predict(i + x + 2)[0].item() - fprices[i])
        sum = sum + (svr_rbf.predict(i + x + 2)[0].item() - fprices[i])**2
    r = (sum / 3) ** 0.5 # standard deviation
    M = M / 3 # expectation
    
    return svr_rbf.predict(x)[0], r, M, svr_rbf.predict(x + 2)[0], fprices[0]

dates, prices, i = get_data('UsdRubLight.csv')
fdates, fprices, fi = get_data('UsdRubLightFuture.csv')
predicted_price = predict_prices(dates, prices, i, fdates, fprices)
print(i)
print(predicted_price)