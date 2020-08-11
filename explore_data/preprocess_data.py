from sklearn.preprocessing import MinMaxScaler



# X (sample, features), y numpy array
# return X: (samples, timesteps, features)
#        y: expected values
def create_dataset(X, y, time_steps=1):
    features = X.shape[1]
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i : i+time_steps]
        Xs.append(v)
        ys.append(y[i + time_steps])
    Xs, ys = np.array(Xs), np.array(ys)
    return Xs, ys


def scale(dataset):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(dataset)
    scaled_dataset = scaler.transform(dataset)
    return scaler, scaled_dataset


def invert_scale(scaler, data):
    return scaler.invert_scale(data)


def split(X, y, train_size=0.8):
    n_train = int(len(X) * train_size)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, y_train, X_test, y_test