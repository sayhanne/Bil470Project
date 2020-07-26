import pandas as pd

class DataPrep:
    """"""
    def __init__(self):
        data = pd.read_csv("FinancialForecast\\Data\\BIST_100_Gecmis_Verileri_Haftalik.csv")
        data_ex = pd.read_csv("FinancialForecast\\Data\\BIST_100_RSI.csv")
        train_x_arr, train_y_arr = [], []
        test_x_arr, test_y_arr = [], []
        columnnames = []
        for i in range(1, 15):
            columnnames.append("Hafta" + str(i))
        columnnames.extend(list(data.columns[8:]))
        test = False

        for row_index in range(data.shape[0] - 1):
            row_now = data.iloc[row_index]
            target = data["Fark %"].iloc[row_index + 1]
            piece_now = None
            
            #onceki 13 gunu ekler
            if row_index < 13:
                piece_now = list(data_ex["Fark %"].iloc[row_index + 2:])
                piece_now.extend(list(data["Fark %"].iloc[:row_index]))
            else:
                piece_now = list(data["Fark %"].iloc[row_index - 13 : row_index])
            row_now = data.iloc[row_index]
            piece_now.append(row_now.iloc[6])
            piece_now.extend(list(row_now.iloc[8:]))
            
            if test:
                if row_index % 28 == 27:
                    test = False
                elif row_index % 28 > row_index % 14:
                    continue
                else:
                    test_x_arr.append(piece_now)
                    test_y_arr.append(target)
            else:
                if row_index % 28 == 27:
                    test = True
                elif row_index % 28 > row_index % 14:
                    continue
                else:
                    train_x_arr.append(piece_now)
                    train_y_arr.append(target)
        # print(train_x_arr)


        train_y_arr_class = []
        for elem in train_y_arr:
            if elem > 0:
                train_y_arr_class.append(1)
            else:
                train_y_arr_class.append(0)

        test_y_arr_class = []
        for elem in test_y_arr:
            if elem > 0:
                test_y_arr_class.append(1)
            else:
                test_y_arr_class.append(0)

        train_x = pd.DataFrame(train_x_arr, columns=columnnames).reset_index().drop(columns="index")
        train_x.to_csv('NeuralNetwork\\DataForTF\\train_x.csv', index=False)
        train_y = pd.DataFrame(train_y_arr, columns=["Target"]).reset_index().drop(columns="index")
        train_y.to_csv('NeuralNetwork\\DataForTF\\train_y.csv', index=False)
        train_y_class = pd.DataFrame(train_y_arr_class, columns=["Target"]).reset_index().drop(columns="index")
        train_y_class.to_csv('NeuralNetwork\\DataForTF\\train_y_class.csv', index=False)
        ###############
        test_x = pd.DataFrame(test_x_arr, columns=columnnames).reset_index().drop(columns="index")
        test_x.to_csv('NeuralNetwork\\DataForTF\\test_x.csv', index=False)
        test_y = pd.DataFrame(test_y_arr, columns=["Target"]).reset_index().drop(columns="index")
        test_y.to_csv('NeuralNetwork\\DataForTF\\test_y.csv', index=False)
        test_y_class = pd.DataFrame(test_y_arr_class, columns=["Target"]).reset_index().drop(columns="index")
        test_y_class.to_csv('NeuralNetwork\\DataForTF\\test_y_class.csv', index=False)
        # print(train_y)

    
DataPrep()
