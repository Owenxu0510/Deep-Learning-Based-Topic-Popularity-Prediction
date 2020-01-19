import jieba
import jieba.posseg as psg
import codecs
import xlrd
import numpy as np
from sklearn import svm
import random
from sklearn.externals import joblib
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Convolution1D
from keras.layers import Activation
from keras.layers import MaxPool1D
from keras.utils import np_utils
from keras.models import load_model
from keras.layers import GRU
from keras.layers import Bidirectional

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score


SVM_kernel = 'rbf'
SVM_gamma = 10

workbook = xlrd.open_workbook('newcontent1.xls')

sheetname = workbook.sheet_names()
sheet1 = workbook.sheet_by_name(sheetname[1])
sheet2 = workbook.sheet_by_name(sheetname[2])

def get_train():
    total_row = sheet1.nrows
    input = []
    input_final = []
    output = []
    for each in range(total_row):
        input1 = []
        if each == 0:
            continue
        x1 = float(sheet1.cell(each, 12).value)
        x2 = float(sheet1.cell(each, 13).value)
        x3 = float(sheet1.cell(each, 14).value)
        x4 = float(sheet1.cell(each, 15).value)
        y = int(sheet1.cell(each, 16).value)

        input1.append(x1)
        input1.append(x2)
        input1.append(x3)
        input1.append(x4)
        input1.append(y)
        input.append(input1)
    random.shuffle(input)

    for each in input:
        x = each[:4]
        input_final.append(x)
        output.append(each[4])

    return input_final, output


def get_test():
    total_row = sheet2.nrows
    input = []
    output = []
    for each in range(total_row):
        input1 = []
        if each == 0:
            continue
        x1 = float(sheet2.cell(each, 12).value)
        x2 = float(sheet2.cell(each, 13).value)
        x3 = float(sheet2.cell(each, 14).value)
        x4 = float(sheet2.cell(each, 15).value)
        y = int(sheet2.cell(each, 16).value)

        input1.append(x1)
        input1.append(x2)
        input1.append(x3)
        input1.append(x4)
        input.append(input1)
        output.append(y)
    return input, output

def SVM_predict(x, y, x1, x2):
    x = np.array(x)
    y = np.array(y)
    nsamples, nx = x.shape
    #nsamples, nx, ny = x.shape
    xx = x.reshape((nsamples, nx))
    forecast = []
    clf = svm.SVC(C=0.8, kernel=SVM_kernel, gamma=SVM_gamma, decision_function_shape='ovr')
    clf.fit(xx, y)
    print("train:", clf.score(x1, x2))
    return clf.predict(x1)
    # print("train:", clf.score(xx, y))

if __name__ == "__main__":
    train_in, train_out = get_train()
    test_in, test_out = get_test()
    x = SVM_predict(train_in, train_out, test_in, test_out)
    print(x)
