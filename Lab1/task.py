import math


class ExpressionExecuter:
    @staticmethod
    def execute():
        x = input("Введите x: ")
        result = ExpressionExecuter.__execute(x)
        print("результат вычисления = " + str(result))

    @staticmethod
    def __handleInput(input):
        input = str(input)
        word = list(filter(lambda wrd: wrd != '', input.split(' ')))[0]

        try:
            x = float(word)
            return x
        except Exception:
            raise Exception("number expected")

    @staticmethod
    def __execute(input):
        try:
            x = ExpressionExecuter.__handleInput(input)
        except Exception as e:
            print(repr(e))
            return

        return math.sin(x) + x ** 3 + 1 / (x ** 2 + 1)


class OddFinder:
    @staticmethod
    def execute():
        x = input("Введите целые числа, среди которых нужно найти нечетные через пробел: ")
        result = OddFinder.__find(x)
        print("Количество нечетных чисел = " + str(result))

    @staticmethod
    def __handleInput(input):
        input = str(input)
        words = list(filter(lambda wrd: wrd != '', input.split(' ')))
        numbers = []
        try:
            for word in words:
                number = int(word)
                numbers.append(number)
            return numbers
        except ValueError:
            raise Exception("integers expected")

    @staticmethod
    def __find(input):
        try:
            numbers = OddFinder.__handleInput(input)
        except Exception as e:
            print(repr(e))
            return
        filteredNumbers = list(filter(lambda number: number % 2 != 0, numbers))
        return len(filteredNumbers)


class WordSorter:
    @staticmethod
    def execute():
        x = input("Введите слова через пробел: ")
        result = WordSorter.__sortWords(x)
        print("Отсортировано по длине: " + str(result))

    @staticmethod
    def __handleInput(input):
        return list(filter(lambda wrd: wrd != '', input.split(' ')))

    @staticmethod
    def __sortWords(input):
        words = WordSorter.__handleInput(input)
        words.sort(key=lambda word: len(word))
        return words
