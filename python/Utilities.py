"""
Created on Sun Apr 14 19:05:12 2013

@author1: Mohamed Aly <mohamed@mohamedaly.info>
@author2: Mahmoud Nabil <mah.nabil@yahoo.com>

"""

import numpy as np
from symbol import return_stmt
import matplotlib.pyplot as plt
import codecs
import re

def show_most_informative_features(vectorizer, clf, file_name, n=20):
    out_file = open(file_name + "_" + str(n) + ".txt", 'w', buffering=100)
    c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
    top = zip(c_f[:n], c_f[:-(n + 1):-1])
    for (c1, f1), (c2, f2) in top:
        line = ("%-15s\n" % (f1))
        line =re.sub(("^\s+"), "", line)
        line =re.sub(("\s+$"), "", line)
        out_file.write(line+'\n')
    for (c1, f1), (c2, f2) in top:
        line = ("%-15s\n" % ( f2))
        line =re.sub(("^\s+"), "", line)
        line =re.sub(("\s+$"), "", line)
        out_file.write(line+'\n')
def MySelectPercentile(vectorizer,feat_ext, precent, X_train, y_train, X_test):
    name=feat_ext['name']
    feat_ext=feat_ext['feat_ext']
    
    # fit the classifier
    feat_ext.fit(X_train, y_train) 
    # total number of features
    N = len(feat_ext.coef_[0]) 
    # sort the coeffiecints in descending order
    c_f = sorted(zip(feat_ext.coef_[0], range(0, N)), reverse=True) 
    
    # get indicies of the features in the sorted order    
    Indeces = [x[1] for x in c_f] 
    # precentage of features to be selected
    N_selected = int (round((precent / 100.0) * N))
    
#     show_most_informative_features(vectorizer, feat_ext, name, N_selected)
    
    print ("N_selected : ", N_selected)
    
    Indeces_Selected_positive = Indeces[:N_selected / 2]
    Indeces_Selected_negative = Indeces[N_selected / (-2):]
    
    Indeces_Selected = np.concatenate((Indeces_Selected_positive, Indeces_Selected_negative))
    X_train_modified = X_train[:, Indeces_Selected]
    X_test_modified = X_test[:, Indeces_Selected]
    return X_train_modified, X_test_modified
    

def Evaluate_Result(pred, y_test):
    # Weighted average of accuracy and f1
    (acc, tacc, support, f1) = (list(), list(), list(), list())
    for l in np.unique(y_test):
        support.append(np.sum(y_test == l) / float(y_test.size))
        tp = float(np.sum(pred[y_test == l] == l))
        fp = float(np.sum(pred[y_test != l] == l))
        fn = float(np.sum(pred[y_test == l] != l))
        #print("tp:", tp, " fp:", fp, " fn:", fn,"class:",l,"precision:",tp/(tp+fp),"recall:",tp/(tp+fn))
        if tp > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
        else:
            (prec, rec) = (0, 1)
        
        f1.append(2 * prec * rec / (prec + rec))
        acc.append(tp / float(np.sum(y_test == l)))
        tacc.append(tp)
                        
    # compute total accuracy
    tacc = np.sum(tacc) / y_test.size
    # weighted accuracy
    acc = np.average(acc, weights=support)
    # weighted F1 measure
    f1 = np.average(f1, weights=support)
            
    print("f1 = %0.3f" % f1)
    print("wacc = %0.3f" % acc)
    print("tacc = %0.3f" % tacc)  
    return (acc, tacc, support, f1)

def Train_And_Predict(X_train, y_train, X_test, classifier, classifier_name):
    
####################################Training########################################                        
    print("Training: ", classifier_name)
    classifier.fit(X_train, y_train)
####################################Testing#########################################
    print("Testing")              
    # for knn predict patches of patterns to save memory
    if(classifier_name == 'KNN'):         
        n = X_test.shape[0]
        patch_size = 100
        div = n / patch_size
        pred = np.array([])
        for  i in range (0, div):
            X_test_patch = X_test[(i * patch_size):(((i + 1) * patch_size)), :]
            pred_patch = classifier.predict(X_test_patch)
            pred = np.concatenate((pred, pred_patch))
        if (div * patch_size < n):
            X_test_patch = X_test[div * patch_size:n, :]
            pred_patch = classifier.predict(X_test_patch)
            pred = np.concatenate((pred, pred_patch))
    else:
        pred = classifier.predict(X_test)
    return pred

def plot(accuracies, precentages, legend_names, feat_extract):
    fig = plt.figure()
    subplot = fig.add_subplot(111)
    color_ = ["g", "b", "r", "c", "m", "y", "b"]
    for i in range(0, accuracies.shape[0]):
        subplot.plot(range(1,len(precentages)+1), accuracies[i,:], color=color_[i],
                marker="D", label=legend_names[i])
#     subplot.plot(range(1,len(precentages)+1), accuracies, color=color_[0],
#                 marker="D", label=legend_names)
    plt.xlabel("Precentage of Features")
    
    plt.xticks(range(1,len(precentages)+1))
    plt.gca().set_xticklabels(precentages)

    
    plt.ylabel("F-Measure")
    plt.title("F1 measure for < " + feat_extract + " > vs Precentage of Features")

    # #set legend box
    box = subplot.get_position()
    subplot.set_position([box.x0, box.y0 + box.height * 0.3,
                     box.width, box.height * 0.7])
    subplot.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True, ncol=3)
    
    plt.draw()
    


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


    
def ReadLexicon():
    pos = codecs.open('../data/lexicon/Pos.txt', 'r', 'utf-8').readlines()
    neg = codecs.open('../data/lexicon/Neg.txt', 'r', 'utf-8').readlines()
    for i in range(0,len(pos)):
        pos[i]=re.sub('\s$','',pos[i])
    for i in range(0,len(neg)):
        neg[i]=re.sub('\s$','',neg[i])        
    lexicon=pos+neg
    return lexicon
def ReadLexicon1():
    pos = codecs.open('../data/sam_lex/Pos.txt', 'r', 'utf-8').readlines()
    neg = codecs.open('../data/sam_lex/Neg.txt', 'r', 'utf-8').readlines()
    for i in range(0,len(pos)):
        pos[i]=re.sub('\s$','',pos[i])
    for i in range(0,len(neg)):
        neg[i]=re.sub('\s$','',neg[i])        
    lexicon=pos+neg
    return lexicon