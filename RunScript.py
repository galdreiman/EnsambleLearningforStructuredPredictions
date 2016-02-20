<<<<<<< HEAD
import inspect
from EnsambleLearningforStructuredPredictions import EnsambleLearningforStructuredPredictions
from numpy import genfromtxt, savetxt
import csv
import scipy.spatial.distance as distance
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import label_binarize
import numpy
import timeit
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp




#start run: 12:15 9/2/2016
class RunScript():

    def __init__(self):
        self.classes = dict()

        self.classes['letter'] = {0:'a',1:'m',2:'n',3:'d',4:'i',5:'o',6:'g'}
        self.classes['ocr-data-test'] = {0:'a',1:'m',2:'n',3:'d',4:'i',5:'o',6:'g'}
        self.classes['ContraceptiveMethodChoice'] = {0:1, 1:2,2:3}
        self.classes['heart'] = {x:str(x+1) for x in range(3)}
        self.classes['abalone-db'] = {x:str(x+1) for x in range(30)}
        self.classes['SatLog-Shuttle'] = {x:str(x+1) for x in range(8)}
        self.classes['SatLog-landset'] = {x:str(x+1) for x in range(8)}
        self.classes['SatLog-ImageSegmentation'] = {x:str(x+1) for x in range(8)}
        self.classes['cmc'] = {x:str(x+1) for x in range(4)}
        self.classes['dermatology'] = {x:str(x+1) for x in range(6)}

        # self.classes['SatLog-landset'] = {}
        # self.classes['SatLog-landset'] = {}

    def run_SVM(self,filename):
        elapssed_time = 0
        mse = 0
        start = timeit.default_timer()

        # 1 - read the data, create X,Y
        print("start readCsv(train)")
        train_X, train_Y, test_X, test_Y = self.readCsv("Data/{}.csv".format(filename))

        # 2 - fit
        print("start FIT . . .")
        clf = RandomForestClassifier()
        clf.fit(train_X, train_Y)
        print "end FIT successfully"

        # 3 - predict
        print "start PREDICT . . ."
        predicted_y = clf.predict(test_X)
        print "end PREDICT successfully"

        stop = timeit.default_timer()

        # print("predicted = {}".format(predicted_y))
        hamming_dist = distance.hamming(predicted_y, test_Y)
        try:
            mse =mean_squared_error(predicted_y, test_Y)
        except Exception :
            mse = -1
        elapssed_time = stop -start
        print("Hamming Distance = {}".format(hamming_dist))

        return hamming_dist,mse, elapssed_time

    def run(self,beta,L,filename):
        elapssed_time = 0
        mse = 0
        start = timeit.default_timer()

        # 1 - read the data, create X,Y
        print("start readCsv(train)")
        train_X, train_Y, test_X, test_Y = self.readCsv("Data/{}.csv".format(filename))

        # 2 - fit
        print("start FIT . . .")
        clf = EnsambleLearningforStructuredPredictions(beta=beta, L=L)
        clf.fit(train_X, train_Y)
        print "end FIT successfully"

        # 3 - predict
        print "start PREDICT . . ."
        predicted_y = clf.predict(test_X)
        proba_y = clf.predict_proba(test_X)
        print "end PREDICT successfully"

        # 4 - plot
        fpr, tpr, roc_auc, n_classes = self.calc_ROC_curve(test_Y, proba_y,filename)

        stop = timeit.default_timer()

        # print("predicted = {}".format(predicted_y))
        hamming_dist = distance.hamming(predicted_y, test_Y)
        try:
            mse =mean_squared_error(predicted_y, test_Y)
        except Exception :
            mse = -1
        elapssed_time = stop -start
        print("Hamming Distance = {}".format(hamming_dist))

        return hamming_dist, mse, elapssed_time, fpr, tpr, roc_auc, n_classes

    def readCsv(self,filename):
        train_X = []
        train_Y = []
        test_X = []
        test_Y = []
        with open(filename, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                X = [float(x) for x in row[4:-1]]
                y = row[-1]
                if random.random() > 0.7:
                    train_X.append(X)
                    train_Y.append(y)
                else:
                    test_X.append(X)
                    test_Y.append(y)
            return train_X, train_Y, test_X, test_Y

    def calc_ROC_curve(self,y_test, y_proba,filename):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_test = label_binarize(y_test, classes=self.classes[filename].values())
        n_classes = y_test.shape[1]
        r = len(self.classes[filename].values())
        for i in range(r):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_proba)
            roc_auc[i] = auc(fpr[i], tpr[i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        return fpr, tpr, roc_auc, n_classes


    def create_plot(self,fpr, tpr, roc_auc,n_classes,beta, L,dataset_name):
        plt.figure()

        ########################################
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve for label {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        #########################################

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('BETA = {0}  L = {1}'.format(beta,L))
        plt.legend(loc="lower right",prop={'size':6})
        plt.savefig("Results/fig_beta_{0}_-{1}_L-{2}.png".format(beta,dataset_name,L),dpi=900)
        # plt.show()

    def create_plot_old(self,fpr, tpr, roc_auc,n_classes,beta, L,dataset_name):
        #########################################
        mean_fpr = [0]*(len(fpr[0])+1)
        mean_fpr_counter = [0]*(len(fpr[0])+1)

        for row_index in range(len(fpr)):
            for col_index in range(len(fpr[row_index])):
                try:
                    mean_fpr[col_index] += fpr[row_index][col_index]
                    mean_fpr_counter[col_index] += 1
                except:
                    print "ERROR"
                    print 'row_index ' + str(row_index)
                    print 'col_index ' + str(col_index)
                    print 'n_classes ' + str(n_classes)
                    print 'len(fpr) ' + str(len(fpr))

        for i in range(len(mean_fpr)-1):
            mean_fpr[i] = mean_fpr[i]/mean_fpr_counter[i]

        #########################################
        mean_tpr = [0]*(len(tpr[0])+1)
        mean_tpr_counter = [0]*(len(tpr[0])+1)

        for row_index in range(len(tpr)):
            for col_index in range(len(tpr[row_index])):
                try:
                    if str(tpr[row_index][col_index]) == 'nan':
                        continue
                    mean_tpr[col_index] += tpr[row_index][col_index]
                    mean_tpr_counter[col_index] += 1
                except:
                    print "ERROR"
                    print 'row_index ' + str(row_index)
                    print 'col_index ' + str(col_index)
                    print 'n_classes ' + str(n_classes)
                    print 'len(tpr) ' + str(len(tpr))

        for i in range(len(mean_tpr)-1):
            mean_tpr[i] = mean_tpr[i]/mean_tpr_counter[i]

        #########################################
        mean_roc_auc = 0
        mean_roc_auc_counter = 0
        for v in roc_auc.values():
            if str(v) == 'nan':
                continue
            mean_roc_auc += v
            mean_roc_auc_counter += 1
        mean_roc_auc /= mean_roc_auc_counter

        #########################################
        plt.figure()
        plt.plot(mean_fpr[:-1], mean_tpr[:-1], label='ROC curve (area = {0})'.format( str(mean_roc_auc)))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('BETA = {0}  L = {1}'.format(beta,L))
        plt.legend(loc="lower right",prop={'size':6})
        plt.savefig("Results/fig_beta_{0}_-{1}_L-{2}.png".format(beta,dataset_name,L),dpi=900)
        # plt.show()

    def run_experiment(self, dataset_name):
        Table = []
        Table.append(['alg name','beta', 'L', 'hamming_dist','mse', 'elapssed_time'])
        # 1- run elsp with L=2,4,6 beta=0.3,0.4,0.5
        for beta in numpy.arange(0.7,1,0.1):
            for L in range(2,9,2):
                try:
                    hamming_dist,mse, elapssed_time, fpr, tpr, roc_auc, n_classes = self.run(beta,L,dataset_name)
                except:
                    continue
                self.create_plot(fpr, tpr, roc_auc,n_classes,beta,L,dataset_name)
                Table.append(['EnsambleLearningforStructuredPredictions',beta,L,hamming_dist,mse,elapssed_time])

        # 2- run svm for comparison:
        hamming_dist,mse, elapssed_time = self.run_SVM(dataset_name)
        Table.append(['SVM',-1,-1,hamming_dist,mse,elapssed_time])

        # 3- write resaults to file
        with open('Results/results_{0}.csv'.format(dataset_name),'wb')as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in Table:
                spamwriter.writerow(row)


if __name__ == '__main__':
    r = RunScript()
    # datasets = ['letter','ContraceptiveMethodChoice','heart','abalone-db','SatLog-Shuttle','SatLog-landset','SatLog-ImageSegmentation']
    datasets = ['dermatology']
    for dataset in datasets:
        r.run_experiment(dataset)

=======

from EnsambleLearningforStructuredPredictions import EnsambleLearningforStructuredPredictions
from numpy import genfromtxt, savetxt
import csv
import scipy.spatial.distance as distance
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import label_binarize
import numpy
import timeit
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp




#start run: 12:15 9/2/2016
class RunScript():

    def __init__(self):
        self.classes = dict()

        self.classes['letter'] = {0:'a',1:'m',2:'n',3:'d',4:'i',5:'o',6:'g'}
        self.classes['ocr-data-test'] = {0:'a',1:'m',2:'n',3:'d',4:'i',5:'o',6:'g'}
        self.classes['ContraceptiveMethodChoice'] = {0:1, 1:2,2:3}
        self.classes['heart'] = {x:str(x+1) for x in range(3)}
        self.classes['abalone-db'] = {x:str(x+1) for x in range(30)}
        self.classes['SatLog-Shuttle'] = {x:str(x+1) for x in range(8)}
        self.classes['SatLog-landset'] = {x:str(x+1) for x in range(8)}
        self.classes['SatLog-ImageSegmentation'] = {x:str(x+1) for x in range(8)}

        # self.classes['SatLog-landset'] = {}
        # self.classes['SatLog-landset'] = {}

    def run_SVM(self,filename):
        elapssed_time = 0
        mse = 0
        start = timeit.default_timer()

        # 1 - read the data, create X,Y
        print("start readCsv(train)")
        train_X, train_Y, test_X, test_Y = self.readCsv("Data/{}.csv".format(filename))

        # 2 - fit
        print("start FIT . . .")
        clf = RandomForestClassifier()
        clf.fit(train_X, train_Y)
        print "end FIT successfully"

        # 3 - predict
        print "start PREDICT . . ."
        predicted_y = clf.predict(test_X)
        print "end PREDICT successfully"

        stop = timeit.default_timer()

        # print("predicted = {}".format(predicted_y))
        hamming_dist = distance.hamming(predicted_y, test_Y)
        try:
            mse =mean_squared_error(predicted_y, test_Y)
        except Exception :
            mse = -1
        elapssed_time = stop -start
        print("Hamming Distance = {}".format(hamming_dist))

        return hamming_dist,mse, elapssed_time

    def run(self,beta,L,filename):
        elapssed_time = 0
        mse = 0
        start = timeit.default_timer()

        # 1 - read the data, create X,Y
        print("start readCsv(train)")
        train_X, train_Y, test_X, test_Y = self.readCsv("Data/{}.csv".format(filename))

        # 2 - fit
        print("start FIT . . .")
        clf = EnsambleLearningforStructuredPredictions(beta=beta, L=L)
        clf.fit(train_X, train_Y)
        print "end FIT successfully"

        # 3 - predict
        print "start PREDICT . . ."
        predicted_y = clf.predict(test_X)
        proba_y = clf.predict_proba(test_X)
        print "end PREDICT successfully"

        # 4 - plot
        fpr, tpr, roc_auc = self.calc_ROC_curve(test_Y, proba_y,filename)

        stop = timeit.default_timer()

        # print("predicted = {}".format(predicted_y))
        hamming_dist = distance.hamming(predicted_y, test_Y)
        try:
            mse =mean_squared_error(predicted_y, test_Y)
        except Exception :
            mse = -1
        elapssed_time = stop -start
        print("Hamming Distance = {}".format(hamming_dist))

        return hamming_dist, mse, elapssed_time, fpr, tpr, roc_auc

    def readCsv(self,filename):
        train_X = []
        train_Y = []
        test_X = []
        test_Y = []
        with open(filename, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spamreader:
                X = [float(x) for x in row[4:-1]]
                y = row[-1]
                if random.random() > 0.7:
                    train_X.append(X)
                    train_Y.append(y)
                else:
                    test_X.append(X)
                    test_Y.append(y)
            return train_X, train_Y, test_X, test_Y

    def calc_ROC_curve(self,y_test, y_proba,filename):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        y_test = label_binarize(y_test, classes=self.classes[filename].values())
        r = len(self.classes[filename].values())
        for i in range(r):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_proba)
            roc_auc[i] = auc(fpr[i], tpr[i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        return fpr, tpr, roc_auc


    def create_plot(self,fpr, tpr, roc_auc,beta, L,dataset_name):
        plt.figure()
        for i in range(len(fpr)-1):
            plt.plot(fpr[i], tpr[i], label="ROC curve for Label {0}  AUC = {1}".format(str(self.classes[dataset_name][i]),str(roc_auc[i])))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('BETA = {0}  L = {1}'.format(beta,L))
        plt.legend(loc="lower right",prop={'size':6})
        plt.savefig("Results/fig_beta_{0}_-{1}_L-{2}.png".format(beta,dataset_name,L),dpi=900)
        # plt.show()

    def run_experiment(self, dataset_name):
        Table = []
        Table.append(['alg name','beta', 'L', 'hamming_dist','mse', 'elapssed_time'])
        # 1- run elsp with L=2,4,6 beta=0.3,0.4,0.5
        for beta in numpy.arange(0.7,1,0.1):
            for L in range(2,9,2):
                try:
                    hamming_dist,mse, elapssed_time, fpr, tpr, roc_auc = self.run(beta,L,dataset_name)
                except:
                    continue
                self.create_plot(fpr, tpr, roc_auc,beta,L,dataset_name)
                Table.append(['EnsambleLearningforStructuredPredictions',beta,L,hamming_dist,mse,elapssed_time])

        # 2- run svm for comparison:
        hamming_dist,mse, elapssed_time = self.run_SVM(dataset_name)
        Table.append(['SVM',-1,-1,hamming_dist,mse,elapssed_time])

        # 3- write resaults to file
        with open('Results/results_{0}.csv'.format(dataset_name),'wb')as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in Table:
                spamwriter.writerow(row)


if __name__ == '__main__':
    r = RunScript()
    datasets = ['letter','ContraceptiveMethodChoice','heart','abalone-db','SatLog-Shuttle','SatLog-landset','SatLog-ImageSegmentation']
    for dataset in datasets:
        r.run_experiment(dataset)

>>>>>>> origin/master
