import numpy
import pandas as pd
import numpy as np
import re
import math
from pathlib import Path
import matplotlib.pyplot as plt
import scipy
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler  # , StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import itertools
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import sklearn.metrics as metrics


def convertStr(s):
    """Convert string to either int or float."""
    try:
        ret = int(s)
    except ValueError:
        # Try float.
        ret = float(s)
    return ret


def getNumbersFromStr(haystack):
    return re.findall(r'\d+', haystack)


def setSettingPandas():
    pd.options.display.max_columns = 11
    pd.options.display.max_rows = 20


def normalizeDF(df):
    numericColumns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col].dtype)]
    normalizedDF = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(df[numericColumns]),
                                columns=numericColumns, index=df.index)

    return normalizedDF


def outlier_by_iqr(x, q25, q75):
    """Проверка на выброс по критерию межквартильного размаха"""
    """Возвращаем 0 если не выброс, -1 если выброс ниже, 1 если выше"""
    x = float(x)
    q1 = q25
    q3 = q75
    iqr = q3 - q1
    high = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr

    result = 0
    if x > high:
        return 1
    if x < low:
        return -1
    return result


def outlierSmooth(df, result_column, nonResult_columns):
    for column in df.columns:
        if column == result_column:
            continue
        q1 = df[column].quantile(0.05, interpolation="midpoint")
        q25 = df[column].quantile(0.25, interpolation="midpoint")
        q75 = df[column].quantile(0.75, interpolation="midpoint")
        q99 = df[column].quantile(0.9, interpolation="midpoint")
        count = 0
        for index, value in df[column].items():
            outliarType = outlier_by_iqr(value, q25, q75)
            if outliarType == -1:
                df.at[index, column] = q1
                count += 1
            if outliarType == 1:
                df.at[index, column] = q99
                count += 1
        outliarPercent = count / len(df[column]) * 100
        print("{0}: {1:2f}".format(column, outliarPercent))

        if outliarPercent > 5:
            df.drop(column, axis=1, inplace=True)
            nonResult_columns.remove(column)
    return df


def findBestCombinationCols(df, nonResult_columns, result_column):
    R2 = -1000
    bestFactCols = []
    for l in range(1, len(nonResult_columns) + 1):
        for factCols in itertools.combinations(nonResult_columns, l):
            factCols = list(factCols)
            X_train, X_test, y_train, y_test = train_test_split(df[factCols],
                                                                df[result_column], test_size=0.3)

            train_index = X_train.index  # индексы обучающей выборки

            df["1"] = 1  # чтобы был свободный член, нужен столбец единиц, добавим его
            # в качестве матрицы факторных признаков используем датафрейм с нужными колонками
            X_matr = df.loc[train_index, ["1"] + factCols]
            y_vec = df.loc[train_index, result_column]

            model_linear = sm.OLS(y_vec, X_matr).fit()  # построение модели
            currR2 = model_linear.rsquared
            if currR2 > R2:
                R2 = currR2
                bestFactCols = factCols

    print("Best cols : {0},\n R2 is {1}".format(bestFactCols, R2))
    return (bestFactCols, R2)


def get_regression_metrics(y_test, y_pred, needround=False, ndigits=3):
    """Вычисляет метрики регрессии и возвращает словарь их значений"""
    round0 = lambda x: round(x, ndigits) if needround else x
    return {
        "RSS": round0(((y_test - y_pred) ** 2).sum()),
        "R2": round0(metrics.r2_score(y_test, y_pred)),
        "MSE": round0(metrics.mean_squared_error(y_test, y_pred)),
        "RMSE": round0(metrics.mean_squared_error(y_test, y_pred) ** 0.5),
        "MAE": round0(metrics.mean_absolute_error(y_test, y_pred)),
        "MAPE": round0(metrics.mean_absolute_percentage_error(y_test, y_pred) * 100)
    }


def execute():
    setSettingPandas()
    path = Path("Lab5", "DataSet", "hf.csv")
    wtquality_df = pd.read_csv(str(path), sep=",")

    # Целевые и нецелевые признаки
    # Прогнозируем пригодность воды к употреблению по ряду признаков состава воды
    result_column = "DEATH_EVENT"
    nonResult_columns = [column for column in wtquality_df.columns if column != result_column]
    factCols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'high_blood_pressure', 'serum_creatinine',
                'sex', 'time']
    # (factCols, R2) = findBestCombinationCols(wtquality_df, nonResult_columns, result_column)
    # print(factCols, "\n", R2)

    print(wtquality_df.info())
    print("\n\n{0}".format(wtquality_df.isnull().sum()))

    # print("\n\nРабота с выбросами:\n")
    # outlierSmooth(wtquality_df, result_column, factCols_columns)

    print(wtquality_df.info())
    # normalizeDF(wtquality_df[factCols_columns]).boxplot()
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(wtquality_df[factCols],
                                                        wtquality_df[result_column], test_size=0.3)

    train_index = X_train.index  # индексы обучающей выборки
    test_index = X_test.index  # индексы тестовой выборки

    wtquality_df["1"] = 1  # чтобы был свободный член, нужен столбец единиц, добавим его
    # в качестве матрицы факторных признаков используем датафрейм с нужными колонками
    X_matr = wtquality_df.loc[train_index, ["1"] + factCols]
    y_vec = wtquality_df.loc[train_index, result_column]

    print("применение МНК вручную:")
    print(np.linalg.inv(X_matr.T.dot(X_matr)).dot(X_matr.T).dot(y_vec))
    print("\n\n")

    model_linear = sm.OLS(y_vec, X_matr).fit()  # построение модели
    print(model_linear.summary())  # вывод подробной информации о модели

    print("\n\nЗначимость критериев:\n{0}\n".format(model_linear.pvalues))
    print("\n\nЗначимость R2:\n{0}\n".format(model_linear.rsquared))
    print("коэффициент детерминации - {0}\n\n".format(model_linear.f_pvalue))

    # построим прогноз на тестовой выборке (колонки те же, на каких обучали)
    y_pred = model_linear.predict(wtquality_df.loc[test_index, ["1"] + factCols])
    # получим также модельные значения на обучающей выборке (для визуализации)
    y_model = model_linear.fittedvalues

    # нарисуем график
    # красивой линии тут не выйдет, потому что, во-первых, регрессия множественная,
    # во-вторых, данные перемешаны, поэтому удовольствуемся scatter plot
    plt.scatter(x=train_index, y=wtquality_df.loc[train_index, result_column],
                label="Обучающая выборка - фактические данные", color="black")
    plt.scatter(x=test_index, y=wtquality_df.loc[test_index, result_column],
                label="Тестовая выборка - фактические данные", color="grey")
    plt.scatter(train_index, y_model,
                label="Обучающая выборка - модельные данные", color="blue")
    plt.scatter(test_index, y_pred,
                label="Тестовая выборка - модельные данные", color="red")
    plt.gca().legend()

    plt.show()

    alphaValues = [(lambda x: math.exp(0.2 * x) - 1)(x) for x in numpy.linspace(0, 20, 200, endpoint=True)][1:]
    print(alphaValues)
    ridgeVals = []
    lassoVals = []

    for alpha in alphaValues:
        model_ridge = Ridge(alpha=alpha).fit(X=wtquality_df.loc[train_index, factCols],
                                             y=wtquality_df.loc[train_index, result_column])
        y_pred = model_ridge.predict(wtquality_df.loc[test_index, factCols])
        res = get_regression_metrics(wtquality_df.loc[test_index, result_column], y_pred, True)["R2"]
        ridgeVals.append(res)

        model_ridge = Lasso(alpha=alpha).fit(X=wtquality_df.loc[train_index, factCols],
                                             y=wtquality_df.loc[train_index, result_column])
        y_pred = model_ridge.predict(wtquality_df.loc[test_index, factCols])
        res = get_regression_metrics(wtquality_df.loc[test_index, result_column], y_pred, True)["R2"]
        lassoVals.append(res)

    plt.scatter(x=alphaValues, y=ridgeVals,
                label="Ridge", color="blue")
    plt.scatter(x=alphaValues, y=lassoVals,
                label="Lasso", color="red")
    plt.gca().legend()

    plt.show()
