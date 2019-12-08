import json
from collections import defaultdict 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
import statsmodels.api as sm
from sklearn import metrics
from sklearn.metrics import make_scorer
#from pyearth import Earth
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import StratifiedKFold
from numpy import zeros, newaxis
from keras import backend
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
import math


NUM_TRAIN_EXAMPLES = 10000
NUM_VALIDATION_EXAMPLE = 1000
NUM_TEST_EXAMPLES = 2000

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def get_time_feature(obj):
    try:
        days = obj.get("days", 0)
    except:
        days = 0
        print("exception in days" , obj)
    try:
        hrs = obj.get("hours", 0)
    except:
        hrs = 0
        print("exception in hours" , obj)
    try:
        mins = obj.get("minutes", 0)
    except:
        mins = 0
        print("exception in minutes" , obj)
    try:
        seconds = obj.get("seconds", 0)
    except:
        seconds = 0
        print("exception in seconds" , obj)
    return (days*24) + (hrs) + (mins/60) + (seconds/60*60)

def convert_queue(kind):
    if kind == "batch":
        return 0
    elif kind == "bigmem":
        return 1
    elif kind == "debug":
        return 2
    elif kind == "long":
        return 3
    elif kind == "short":
        return 4
    elif kind == "phi":
        return 5
    elif kind == "large":
        return 6
    else:
        print("New kind is :", kind)
        return 7

def convert_true_false(mbool):
    if mbool == True:
        return 1
    else:
        return 0

def convert_app_name(appname):
    if appname == 'amber':  return 0
    elif appname == 'charmm':  return 1
    elif appname == 'comsol':  return 2
    elif appname == 'fast':  return 3
    elif appname == 'fluent':  return 4
    elif appname == 'gamess':  return 5
    elif appname == 'gaussian':  return 6
    elif appname == 'gromacs':  return 7
    elif appname == 'ls-dyna':  return 8
    elif appname == 'matlab': return 9
    elif appname == 'mfix': return 10
    elif appname == 'mono': return 11
    elif appname == 'namd':  return 12
    elif appname == 'nek':  return 13
    elif appname == 'python':  return 14
    elif appname == 'quantum_espresso':  return 15
    elif appname == 'vasp': return 16
    elif appname == 'unknown': return 17
    elif appname == 'windplansolver': return 18
    elif appname == 'wrf': return 19
    else: return 20

def convert_feature_request(feature):
    if feature == '!phi': return 0
    elif feature == '24core': return 1
    elif feature == None: return 2
    elif feature == '256GB': return 3
    elif feature == 'anynode': return 4
    elif feature == '!su06': return 5
    elif feature == '64GB': return 6
    elif feature == '16core': return 7
    elif feature == 'phi': return 8
    else:
        print(feature)
        return 9

duration = defaultdict(list)
total_energy = defaultdict(list)
median_powers = defaultdict(list)
power_range = defaultdict(list)
X = []
X_insitu = []
Y=[]
flag = 1
with open("10k.anon.json") as data:
    iterations = -1
    count = 0
    for line in data:
        #if count == 30:
        #    print(line)
        count += 1
        iterations += 1
        if iterations < NUM_TRAIN_EXAMPLES :
            vals = []
            obj = json.loads(line)
            if obj["job"]["torque_exit_code"] == 0:
                #print("[dbg] torque exit code is not 0")
                #continue
                if flag == 2:
                    print(line)
                flag += 1
                size = 0
                num_of_not_valid = 0
                for ts in obj["power"]:
                    try:
                        if ts["metric"] == "power":
                            size += 1
                            if obj["job"]["feature_req"] == "phi" and ts["value"] <= 700 and ts["value"] >= 90: 
                                vals.append(ts["value"])
                            elif obj["job"]["feature_req"] != "phi" and ts["value"] <= 300 and ts["value"] >= 90:
                                vals.append(ts["value"])
                            else:
                                num_of_not_valid += 1
                    except:
                        continue
                if vals == [] or np.var(vals) < 10 or (float(num_of_not_valid) / size) >= 0.5 :
                    continue
                try:
                    duration[obj["job"]["app_name"]].append(get_time_feature(obj["job"]["wallclock_used"]))
                    total_energy[obj["job"]["app_name"]].append(get_time_feature(obj["job"]["wallclock_used"])*np.mean(vals))
                    median_powers[obj["job"]["app_name"]].append(np.median(vals))
                    power_range[obj["job"]["app_name"]].append(np.ptp(vals))
                    X.append( [float(convert_queue(obj["job"]["queue"])),
                    float(convert_app_name(obj["job"]["app_name"])),
                    float(obj["job"]["processors_used"]),
                    float(get_time_feature(obj["job"]["cpu_used"])),
                    #float(obj["job"]["mem_used"]),
                    #float(obj["job"]["vmem_used"]),
                    float(get_time_feature(obj["job"]["wallclock_req"])),
                    float(convert_feature_request(obj["job"]["feature_req"])),
                    float(obj["job"]["nodes_req"]),
                    #float(obj["job"]["processors_req"]),
                    float(convert_true_false(obj["job"]["interactive"])) ] )
                    X_insitu.append( [float(convert_queue(obj["job"]["queue"])),
                    float(convert_app_name(obj["job"]["app_name"])),
                    #float(obj["job"]["processors_used"]),
                    #float(get_time_feature(obj["job"]["cpu_used"])),
                    #float(obj["job"]["mem_used"]),
                    #float(obj["job"]["vmem_used"]),
                    float(get_time_feature(obj["job"]["wallclock_req"])),
                    #float(convert_feature_request(obj["job"]["feature_req"])),
                    #float(obj["job"]["nodes_req"]),
                    #float(obj["job"]["processors_req"]),
                    #float(convert_true_false(obj["job"]["interactive"])) ,
                    np.mean(vals[0:math.ceil(len(vals)/14.0)]) ] )
                    Y.append(np.mean(vals))
                except Exception as e:
                    print("error: " , e)
                
        else:
            break
        
X = preprocessing.MinMaxScaler().fit_transform(X)
X_insitu = preprocessing.MinMaxScaler().fit_transform(X_insitu)

'''X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())

X2 = sm.add_constant(X_insitu)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())'''
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1)

X_insitu_train, X_insitu_test, y_insitu_train, y_insitu_test = train_test_split(X_insitu, Y, test_size = 0.1)

#Linear Regression
reg = LinearRegression(normalize = True, n_jobs = -1)
model = reg.fit(X_train, y_train)
#y_predict = reg.predict(X_test)
#print("root mean squared error is: ", np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
scores = cross_val_score(model, X_train, y_train, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("#####################################################")
print("Linear Regression cross validation score is: ", np.sqrt(scores.mean()))
print("#####################################################")
      
#Linear Regression insitu  
reg = LinearRegression(normalize = True, n_jobs = -1)
model = reg.fit(X_insitu_train, y_insitu_train)
#y_predict = reg.predict(X_test)
#print("root mean squared error is: ", np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
scores = cross_val_score(model, X_insitu_train, y_insitu_train, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("#####################################################")
print("Linear Regression cross validation score is: ", np.sqrt(scores.mean()))
print("#####################################################")
      
#Ridge regression 
clf = Ridge(alpha=1.0)
ridge_model = clf.fit(X_train, y_train)
scores = cross_val_score(ridge_model, X_train, y_train, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("#####################################################")
print("Ridge Regression cross validation score is: ", np.sqrt(scores.mean()))
print("#####################################################")
      
#Ridge regression insitu 
clf = Ridge(alpha=1.0)
ridge_model = clf.fit(X_insitu_train, y_insitu_train)
scores = cross_val_score(ridge_model, X_insitu_train, y_insitu_train, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("#####################################################")
print("Ridge Regression cross validation score is: ", np.sqrt(scores.mean()))
print("#####################################################")
      
#Losso regression
clf = linear_model.Lasso(alpha=0.1)
losso_model = clf.fit(X_train, y_train)
scores = cross_val_score(losso_model, X_train, y_train, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("#####################################################")
print("Losso Regression cross validation score is: ", np.sqrt(scores.mean()))
print("#####################################################")
      
#Losso regression insitu
clf = linear_model.Lasso(alpha=0.1)
losso_model = clf.fit(X_insitu_train, y_insitu_train)
scores = cross_val_score(losso_model, X_insitu_train, y_insitu_train, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("#####################################################")
print("Losso Regression cross validation score is: ", np.sqrt(scores.mean()))
print("#####################################################")

#PLS Regression
pls2 = PLSRegression(n_components=2)
PLS_model = pls2.fit(X, Y)
scores = cross_val_score(PLS_model, X_train, y_train, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("#####################################################")
print("PLS Regression cross validation score is: ", np.sqrt(scores.mean()))
print("#####################################################")
      
      
#PLS Regression insitu
pls2 = PLSRegression(n_components=2)
PLS_model = pls2.fit(X_insitu_train, y_insitu_train)
scores = cross_val_score(PLS_model, X_insitu_train, y_insitu_train, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("#####################################################")
print("PLS Regression cross validation score is: ", np.sqrt(scores.mean()))
print("#####################################################")
      
'''#mars
mars = Earth(max_degree=1, penalty=1.0, endspan=5)
scores = cross_val_score(mars, X, Y, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("mars cross validation score for mars is: ", np.sqrt(scores.mean()))'''

#radom forest
rf = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth=8)
rf_model = rf.fit(X_train,y_train)
scores = cross_val_score(rf_model, X_train, y_train, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("#####################################################")
print("random forest cross validation score is: ", np.sqrt(scores.mean()))
print("#####################################################")
      
#radom forest insitu
rf = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth=8)
rf_model = rf.fit(X_insitu_train,y_insitu_train)
scores = cross_val_score(rf_model, X_insitu_train, y_insitu_train, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("#####################################################")
print("random forest cross validation score is: ", np.sqrt(scores.mean()))
print("#####################################################")

#adaboost
regr = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=8), random_state=0, n_estimators=100, learning_rate = 0.3)
adaboost_model = regr.fit(X_train, y_train)
scores = cross_val_score(adaboost_model, X_train, y_train, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("#####################################################")
print("Adaboost cross validation score is: ", np.sqrt(scores.mean()))
print("#####################################################")
      
#adaboost insitu
regr = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=8), random_state=0, n_estimators=100, learning_rate = 0.3)
adaboost_model = regr.fit(X_insitu_train, y_insitu_train)
scores = cross_val_score(adaboost_model, X_insitu_train, y_insitu_train, scoring=make_scorer(metrics.mean_squared_error), cv = 10)
print("#####################################################")
print("Adaboost cross validation score is: ", np.sqrt(scores.mean()))
print("#####################################################")
      
# create and fit the LSTM network
'''NN_model = Sequential()
NN_model.add(LSTM(4, input_shape=(1, look_back)))
NN_model.add(Dense(1))
NN_model.compile(loss='mean_squared_error', optimizer='adam')
NN_model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)'''
'''seed = 7
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)
cvscores = []'''
X_train = np.array(X_train)
X_train = X_train[:, :, newaxis]
y_train = np.array(y_train)
y_train = y_train[:, newaxis]
X_test = np.array(X_test)
X_test = X_test[:, :, newaxis]
y_test = np.array(y_test)
y_test = y_test[:, newaxis]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#for train, test in kfold.split(X_train, y_train):
    # create model
#    print("train: ", train, "test: ", test)'''
NN_model = Sequential()
NN_model.add(LSTM(1, input_shape=(8,1), recurrent_activation=None, activation=None))
#NN_model.add(Dense(512, activation=None))
NN_model.add(Dense(256, activation=None))
NN_model.add(Dense(128, activation=None))
NN_model.add(Dense(64, activation=None))
NN_model.add(Dense(32, activation=None))
NN_model.add(Dense(16, activation=None))
NN_model.add(Dense(1, activation=None))
# Compile model
NN_model.compile(loss='mse', optimizer='adadelta', metrics=[rmse])
# Fit the model
history = NN_model.fit(X_train, y_train, epochs=50, verbose=1)
# evaluate the model
NN_scores = NN_model.evaluate(X_test, y_test, verbose=1)
print((NN_model.metrics_names[1], NN_scores[1]))
plt.plot(history.history['rmse'])
plt.show()


# create and fit the LSTM network for insitu
'''NN_model = Sequential()
NN_model.add(LSTM(4, input_shape=(1, look_back)))
NN_model.add(Dense(1))
NN_model.compile(loss='mean_squared_error', optimizer='adam')
NN_model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)'''
'''seed = 7
kfold = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)
cvscores = []'''
X_insitu_train = np.array(X_insitu_train)
X_insitu_train = X_insitu_train[:, :, newaxis]
y_insitu_train = np.array(y_insitu_train)
y_insitu_train = y_insitu_train[:, newaxis]
X_insitu_test = np.array(X_insitu_test)
X_insitu_test = X_insitu_test[:, :, newaxis]
y_insitu_test = np.array(y_insitu_test)
y_insitu_test = y_insitu_test[:, newaxis]
print(X_insitu_train.shape)
print(y_insitu_train.shape)
print(X_insitu_test.shape)
print(y_insitu_test.shape)
#for train, test in kfold.split(X_train, y_train):
    # create model
#    print("train: ", train, "test: ", test)'''
NN_model = Sequential()
NN_model.add(LSTM(1, input_shape=(4,1), recurrent_activation=None, activation=None))
#NN_model.add(Dense(512, activation=None))
NN_model.add(Dense(256, activation=None))
NN_model.add(Dense(128, activation=None))
NN_model.add(Dense(64, activation=None))
NN_model.add(Dense(32, activation=None))
NN_model.add(Dense(16, activation=None))
NN_model.add(Dense(1, activation=None))
# Compile model
NN_model.compile(loss='mse', optimizer='adadelta', metrics=[rmse])
# Fit the model
history = NN_model.fit(X_insitu_train, y_insitu_train, epochs=50, verbose=1)
# evaluate the model
#NN_scores = NN_model.evaluate(X_insitu_test, y_insitu_test, verbose=1)
y_pred = NN_model.predict(X_insitu_test, verbose = 1 )
plt.scatter(y_insitu_test, y_pred, c='red')
plt.title('Real Mean Power vs Prediction with Sequential Model')
plt.xlabel('Real Mean Power (W)')
plt.ylabel('Predicted Mean Power (W)')
plt.show()
#print((NN_model.metrics_names[1], NN_scores[1]))
#plt.plot(history.history['rmse'])
#plt.show()

#print("Sequential neural net using LSTM and Dense layers mean cross validation score: ", np.mean(cvscores))
'''clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, Y)
print("Random forest feature importance", clf.feature_importances_)
print("Random forest score: ", clf.score(X, Y))'''

'''plt.boxplot(duration.values(), labels=duration.keys(), widths=0.3)
plt.title("Duration(hours)")
plt.show()

plt.boxplot(total_energy.values(), labels=total_energy.keys(), widths=0.3)
plt.title("Total energy(kWh)")
plt.show()

plt.boxplot(median_powers.values(), labels=median_powers.keys(), widths=0.3)
plt.title("Median power(kW)")
plt.show()

plt.boxplot(power_range.values(), labels=power_range.keys(), widths=0.3)
plt.title("Power range(kW)")
plt.show()'''
