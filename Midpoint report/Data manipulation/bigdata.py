import json
from collections import defaultdict 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

NUM_TRAIN_EXAMPLES = 7000
NUM_VALIDATION_EXAMPLE = 1000
NUM_TEST_EXAMPLES = 2000

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
Y=[]

with open("10k.anon.json") as data:
    iterations = -1
    count = 0
    for line in data:
        if count == 4:
            print(line)
        count += 1
        iterations += 1
        if iterations < NUM_TRAIN_EXAMPLES :
            vals = []
            obj = json.loads(line)
            if obj["job"]["torque_exit_code"] != 0:
                continue
            size = 0
            num_of_not_valid = 0
            for ts in obj["power"]:
                try:
                    if ts["metric"] == "power":
                        size += 1
                        if obj["job"]["queue"] == "phi" and ts["value"] <= 700 and ts["value"] >= 90: 
                            vals.append(ts["value"]/1000)
                        elif obj["job"]["queue"] != "phi" and ts["value"] <= 300 and ts["value"] >= 90:
                            vals.append(ts["value"]/1000)
                        else:
                            num_of_not_valid += 1
                except:
                    continue
            if vals == [] or np.var(vals) == 0 or (num_of_not_valid / size) > 0.5 :
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
                float(obj["job"]["mem_used"]),
                float(obj["job"]["vmem_used"]),
                float(get_time_feature(obj["job"]["wallclock_req"])),
                float(convert_feature_request(obj["job"]["feature_req"])),
                float(obj["job"]["nodes_req"]),
                float(obj["job"]["processors_req"]),
                float(convert_true_false(obj["job"]["interactive"])) ] )
                Y.append(np.mean(vals))
            except Exception as e:
                print("error: " , e)
        else:
            break

'''reg = LinearRegression(normalize = True, n_jobs = -1).fit(X, Y)
print("reg.score(X, Y): " , reg.score(X, Y))
print("reg.coef_ : ", reg.coef_)
print("reg.intercept_ : " , reg.intercept_)'''

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())

plt.boxplot(duration.values(), labels=duration.keys(), widths=0.3)
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
plt.show()

