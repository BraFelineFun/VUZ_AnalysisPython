import pandas as pd
import numpy as np
import re
import random
from pathlib import Path
import matplotlib.pyplot as plt
import scipy
import sklearn.preprocessing


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


def zScoreSeries(series):
    values = series.tolist()
    # Calculate the Standard Deviation in Python
    mean = sum(values) / len(values)
    differences = [(value - mean) ** 2 for value in values]
    sum_of_differences = sum(differences)
    standard_deviation = (sum_of_differences / (len(values) - 1)) ** 0.5
    standard_deviation = 0.01 if standard_deviation == 0 else standard_deviation
    zScores = [(value - mean) / standard_deviation for value in values]
    return pd.Series(zScores)


def normalizeDF(df):
    numericColumns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col].dtype)]
    normalizedDF = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit_transform(df[numericColumns]),
                                columns=numericColumns, index=df.index)

    # normalizedDF2 = pd.DataFrame([zScoreSeries(df[column]) for column in df[numericColumns].columns], columns=df[numericColumns].columns)
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

def createDFGraphic(df):
    plt.rcParams["figure.figsize"] = 12, 8  # ширина, высота

    for index, column in enumerate(df.columns):
        plt.subplot(2, 3, (index+1)*2-1)
        plt.scatter(x=df.index, y=df[column], color="blue")
        plt.ylabel("Значения наблюдений")
        plt.xlabel("Номера наблюдений")
        plt.title("Scatter, {0}".format(column))

        plt.subplot(2, 3, (index+1)*2)
        plt.boxplot(df[column])
        plt.ylabel("Значения показателя")
        plt.title("Box plot, {0}".format(column))

    plt.tight_layout()  # это чтобы надписи друг на друга не наезжали


def createSeriesGraphic(series):
    plt.hist(series, bins=5, density=True,
             color="darkorchid", linewidth=1, edgecolor="black",
             label="Столбцы")
    series.plot.density(color="mediumblue", label="Кривая плотности")

    plt.gca().legend()
    plt.ylabel("Плотность вероятности")  # подпись к оси Y
    plt.xlabel("Границы интервалов")  # подпись к оси X
    plt.title("Гистограмма с наложенной кривой плотности")



def execute():
    setSettingPandas()

    path = Path("Lab4", "DataSet", "loans.csv")
    loans_df = pd.read_csv(str(path), sep=",")
    print(loans_df.info())
    print("=======================================")

    # убираем столбцы, если пропусков > 10%
    print("Убираем столбцы:\n", "before: ", len(loans_df.columns))
    loans_df.dropna(axis="columns", thresh=int(len(loans_df) - len(loans_df) * 0.1), inplace=True)
    print("after: ", len(loans_df.columns))
    print("=======================================")

    # Заполняем пропуски
    for column, nullNumber in loans_df.isna().sum().items():
        if nullNumber == 0:
            continue

        print(column, ": ", nullNumber)
        loans_df[column] = loans_df[column].mode()[0]

    print("Проверим, остались ли пропуски:\n", loans_df.isna().sum())
    print("=======================================")

    # ищем выбросы
    numericCols = [col for col in loans_df.columns if pd.api.types.is_numeric_dtype(loans_df[col].dtype)]
    numericLoans = loans_df[numericCols].copy()
    #normalizedLoans.boxplot()
    #plt.show()

    print("Процент выбросов в каждой колонке")
    for column in numericLoans.columns:
        q1 = numericLoans[column].quantile(0.05, interpolation="midpoint")
        q25 = numericLoans[column].quantile(0.25, interpolation="midpoint")
        q75 = numericLoans[column].quantile(0.75, interpolation="midpoint")
        q99 = numericLoans[column].quantile(0.9, interpolation="midpoint")
        count = 0
        for index, value in numericLoans[column].items():
            outliarType = outlier_by_iqr(value, q25, q75)
            if outliarType == -1:
                numericLoans.at[index, column] = q1
                count += 1
            if outliarType == 1:
                numericLoans.at[index, column] = q99
                count += 1
        outliarPercent = count/len(numericLoans[column])*100
        print("{0}: {1:2f}".format(column, outliarPercent))

        if outliarPercent > 5:
            numericLoans.drop(column, axis=1, inplace=True)

    normalizedLoans = normalizeDF(numericLoans)

    # normalizedLoans.boxplot()
    # plt.show()

    #createDFGraphic(numericLoans)
    #plt.show()


    print("============================\n Задание 2. Работа с колонкой Coapplicant_Income")
    coapIncome_Series = numericLoans["Coapplicant_Income"]
    #createSeriesGraphic(coapIncome_Series)
    #plt.show()

    # протестируем на нормальных данных
    print("Распределение не является нормальным, так как по тесту Шапиро = {0} < "
          "0.05 (alpha)".format(scipy.stats.shapiro(coapIncome_Series)))


    gamma = 0.95 # 95% доверительный интервал
    quantile2 = scipy.stats.t.ppf((gamma + 1) / 2, df=len(coapIncome_Series) - 1)
    # вычислим стандартную ошибку среднего
    se = coapIncome_Series.std() / len(coapIncome_Series) ** 0.5
    # print(f"se={se}, S={X.std()}")

    left = coapIncome_Series.mean() - quantile2 * se  # левая граница интервала
    right = coapIncome_Series.mean() + quantile2 * se  # правая граница интервала

    print(f"С вероятностью {gamma} средний глобальный объем продаж игры составляет [{round(left, 3)}; {round(right, 3)}]")


    left_x = coapIncome_Series.min()
    right_x = coapIncome_Series.max()
    whole_x = np.arange(left_x, right_x, 0.1)
    whole_y = scipy.stats.t.pdf(whole_x, df=len(coapIncome_Series) - 1, loc=coapIncome_Series.mean(), scale=se)
    interval_x = np.arange(left_x, right, 0.1)
    interval_y = scipy.stats.t.pdf(interval_x, df=len(coapIncome_Series) - 1, loc=coapIncome_Series.mean(), scale=se)

    plt.plot(whole_x, whole_y, color="darkblue")
    plt.fill_between(x=interval_x, y1=interval_y, color="blue")
    plt.xticks([left_x, right])
    plt.xlim(left_x, right_x)
    plt.show()


