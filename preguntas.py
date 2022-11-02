
import numpy as np
import pandas as pd


def pregunta_01():

    df = pd.read_csv('gm_2008_region.csv')


    y = df['life']
    X = df['fertility']


    print(y.shape)

 
    print(X.shape)


    y_reshaped = y.values.reshape(-1,1)


    X_reshaped = y.values.reshape(-1,1)


    print(y_reshaped.shape)

    print(X_reshaped.shape)


def pregunta_02():

    df = pd.read_csv('gm_2008_region.csv')

    
    print(df.shape)

    print(round(df['life'].corr(df['fertility']), 4))


    print(round(df['life'].mean(), 4))


    print(type(df['life']))

    print(round(df['GDP'].corr(df['life']), 4))


def pregunta_03():

    df = pd.read_csv('gm_2008_region.csv')

    X_fertility = df['fertility']

    y_life = df['life']

    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()

    prediction_space = np.linspace(
        X_fertility.min(),
        X_fertility.max(),
        X_fertility.shape[0]
    ).reshape(-1, 1)

    X_reshape = X_fertility.values.reshape(-1, 1)
    y_reshape = y_life.values.reshape(-1, 1)

    reg.fit(X_reshape, y_reshape)


    y_pred = reg.predict(prediction_space)

    print(reg.score(X_reshape, y_reshape).round(4))


def pregunta_04():

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    df = pd.read_csv('gm_2008_region.csv')


    X_fertility = df['fertility']

    y_life = df['life']


    (X_train, X_test, y_train, y_test,) = train_test_split(
        X_fertility,
        y_life,
        test_size = 0.2,
        random_state = 53,
    )

    X_train_reshaped = X_train.values.reshape(-1, 1)
    y_train_reshaped = y_train.values.reshape(-1, 1)
    X_test_reshaped = X_test.values.reshape(-1, 1)
    y_test_reshaped = y_test.values.reshape(-1, 1)


    linearRegression = LinearRegression()

    linearRegression.fit(X_train_reshaped, y_train_reshaped)


    y_pred = linearRegression.predict(X_test_reshaped)

    print("R^2: {:6.4f}".format(linearRegression.score(X_test_reshaped, y_test_reshaped)))
    rmse = np.sqrt(mean_squared_error(y_test_reshaped, y_pred))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
