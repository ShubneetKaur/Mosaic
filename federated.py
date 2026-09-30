import numpy as np
from sklearn.metrics import mean_absolute_error

FEATURES = ['hour_sin','hour_cos','is_weekend','weather_index','incident_index','lag_flow']

def predict(X, w, b):
    return X @ w + b

def local_train(X, y, w, b, epochs=20, lr=0.001):
    w = w.copy()
    for _ in range(epochs):
        err = predict(X, w, b) - y
        w -= lr * (2 / len(X)) * (X.T @ err)
        b -= lr * 2 * err.mean()
    return w, float(b), mean_absolute_error(y, predict(X, w, b))

def fedavg(updates):
    total = sum(n for _, _, n, _ in updates)
    w = sum(w*n/total for w, b, n, mae in updates)
    b = sum(b*n/total for w, b, n, mae in updates)
    return w, float(b)

def run_federated(data, rounds=6, epochs=20, lr=0.001):
    d = len(FEATURES)
    w = np.zeros(d)
    b = 0.0
    history=[]
    for r in range(1, rounds+1):
        updates=[]
        sent=0
        for location, df in data.items():
            split=int(len(df)*0.8)
            train=df.iloc[:split]
            X=train[FEATURES].values.astype(float)
            y=train['target_flow'].values.astype(float)
            lw, lb, mae=local_train(X,y,w,b,epochs,lr)
            updates.append((lw,lb,len(train),mae))
            sent += (len(lw)+1)*4
        w,b=fedavg(updates)
        maes=[]
        for df in data.values():
            test=df.iloc[int(len(df)*0.8):]
            maes.append(mean_absolute_error(test['target_flow'], predict(test[FEATURES].values,w,b)))
        history.append({'round':r,'communication_bytes':sent,'global_mae':float(np.mean(maes))})
    return w,b,history
