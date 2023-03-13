import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from umap import UMAP
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import math
import os, shutil

def load_data(data_file):
    """loads the data from the gzip pickled files, and converts to numpy arrays"""
    print('loading data ...')
    f = gzip.open(data_file, 'rb')
    train_set, valid_set, test_set = load_pickle(f)
    f.close()

    train_set_x, train_set_y = make_tensor(train_set)
    valid_set_x, valid_set_y = make_tensor(valid_set)
    test_set_x, test_set_y = make_tensor(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y

def svm_classify_view(outputs_train, outputs_test, C, view = 0):

    labels_t = outputs_train[-1]
    labels_s = outputs_test[-1]


    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(outputs_train[view], labels_t)
    p = clf.predict(outputs_test[view])
    test_acc = accuracy_score(labels_s, p)

    return test_acc



def svm_classify(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, _, train_label = data[0]
    valid_data, _, valid_label = data[1]
    test_data, _, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)


def unpack_data(dataB, device='cuda'):
    # dataB :: (Tensor, Idx) | [(Tensor, Idx)]
    """ Unpacks the data batch object in an appropriate manner to extract data """
    if is_multidata(dataB):
        if torch.is_tensor(dataB[0]):
            if torch.is_tensor(dataB[1]):
                return dataB[0].to(device)  # mnist, svhn, cubI
            elif is_multidata(dataB[1]):
                return dataB[0].to(device), dataB[1][0].to(device)  # cubISft
            else:
                raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[1])))

        elif is_multidata(dataB[0]):
            return [d.to(device) for d in list(zip(*dataB))[0]]  # mnist-svhn, cubIS
        else:
            raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB[0])))
    elif torch.is_tensor(dataB):
        return dataB.to(device)
    else:
        raise RuntimeError('Invalid data format {} -- check your dataloader!'.format(type(dataB)))


def visualize_umap(z,classes, save_file = None):
    z_embed = TSNE().fit_transform(z)

    fig = plt.figure()
    plt.scatter(z_embed[:,0], z_embed[:,1], c=classes)
    plt.legend()
    if save_file is not None:
        plt.savefig(save_file)
    return fig

def save_encoders(model, path):
    if os.path.exists(str(path)+'/model1.pt'):
        shutil.copyfile(str(path)+'/model1.pt', '{}.old'.format(str(path)+'/model1.pt'))
        shutil.copyfile(str(path)+'/model2.pt', '{}.old'.format(str(path)+'/model2.pt'))
    if os.path.exists(str(path)+'/model3.pt'):
        shutil.copyfile(str(path)+'/model3.pt', '{}.old'.format(str(path)+'/model3.pt'))


    torch.save(model.model1.state_dict(),str(path)+'/model1.pt')
    torch.save(model.model2.state_dict(),str(path) + '/model2.pt')
    if hasattr(model, 'model3'): 
        torch.save(model.model3.state_dict(), str(path) + '/model3.pt')

class Constants(object):
    eta = 1e-6
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0