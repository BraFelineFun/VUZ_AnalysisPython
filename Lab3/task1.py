import pandas as pd
import numpy as np
import re
import random


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


def getPlanValues(planSeries):
    planSeries = planSeries.sort_values()
    planValues = {}
    allPlanValues = []
    previousPlan = ""
    previousPlanValue = 0
    seriesLen = len(planSeries)
    maxPlanValue = -1
    for index, plan in enumerate(planSeries.items()):
        currentPlan = plan[1]

        # Так как записи отсортированы, можем пропускать запись, которая идентична предыдущей
        if previousPlan == currentPlan:
            allPlanValues.append(previousPlanValue)
            continue
        # Извлекаем цифру из строки плана
        numSubStr = getNumbersFromStr(currentPlan)
        if len(numSubStr) == 0:
            while True:
                randomPlanValue = random.randint(10, seriesLen)
                if not (randomPlanValue in planValues):
                    planValues[currentPlan] = randomPlanValue
                    previousPlanValue = convertStr(randomPlanValue)
                    break
        else:
            planValues[currentPlan] = numSubStr[0]
            previousPlanValue = convertStr(numSubStr[0])
            if previousPlanValue > maxPlanValue:
                maxPlanValue = previousPlanValue

        allPlanValues.append(previousPlanValue)
        previousPlan = currentPlan
    return allPlanValues, maxPlanValue


def createMaxLine(dataFrame, maxRealPlanValue):
    address = dataFrame.sample(1).squeeze(axis=0).loc["address"]

    maxLine = dataFrame.apply(lambda column: max(column), axis=0)
    maxLine["address"] = address
    maxLine["encodedPlan"] = maxRealPlanValue
    return maxLine


def execute():
    avito_df = pd.read_csv("https://pastebin.com/raw/4Mbv3aYg", sep=",")

    # ====== 1.1 =======
    # Работаем с Plan
    (encodedPlanColumn, maxRealPlanValue) = getPlanValues(avito_df["plan"])
    avito_df["encodedPlan"] = encodedPlanColumn
    # print(avito_df)

    # ====== 1.2 =======
    # Строка с максимальными значениями
    maxLine = createMaxLine(dataFrame=avito_df, maxRealPlanValue=maxRealPlanValue)
    avito_df = pd.concat([avito_df, maxLine.to_frame().T], ignore_index=True)
    # print(avito_df.columns)


    # ====== 1.3 =======
    # фильтрация: площадь квартиры от 50 до 60 м², стоимость от 4 до 5 млн, а этаж минимальный в этой категории.
    avito_df["square"] = avito_df["square"].apply(lambda row: convertStr(getNumbersFromStr(row)[0]))
    avito_df["price"] = avito_df["price"].apply(lambda row: convertStr("".join(getNumbersFromStr(row))))
    avito_df["floor"] = avito_df["floor"].apply(lambda row: convertStr(getNumbersFromStr(row)[0]))
    costAndSquare = avito_df[
        ((avito_df.square >= 50) & (avito_df.square < 60)) &
        ((avito_df.price >= 4000000) & (avito_df.price < 5000000))
        ]
    minFloor = min(avito_df["floor"])
    #print("minimal floor is {0}".format(minFloor))
    costAndSquare = costAndSquare[costAndSquare.floor == minFloor]
    #print(costAndSquare)

    # maskOfSTDandMean = avito_df[
    #     avito_df.square
