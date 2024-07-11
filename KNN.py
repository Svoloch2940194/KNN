'''

Задание 1.1 - Метод К-ближайших соседей (K-neariest neighbor classifier)
В первом задании вы реализуете один из простейших алгоритмов машинного обучения - классификатор на основе метода K-ближайших соседей. Мы применим его к задачам

бинарной классификации (то есть, только двум классам)
многоклассовой классификации (то есть, нескольким классам)
Так как методу необходим гиперпараметр (hyperparameter) - количество соседей, мы выберем его на основе кросс-валидации (cross-validation).

Наша основная задача - научиться пользоваться numpy и представлять вычисления в векторном виде, а также ознакомиться с основными метриками, важными для задачи классификации.

Перед выполнением задания:

запустите файл download_data.sh, чтобы скачать данные, которые мы будем использовать для тренировки
установите все необходимые библиотеки, запустив pip install -r requirements.txt (если раньше не работали с pip, вам сюда - https://pip.pypa.io/en/stable/quickstart/)
Если вы раньше не работали с numpy, вам может помочь tutorial. Например этот:
http://cs231n.github.io/python-numpy-tutorial/

'''


import numpy as np
import matplotlib.pyplot as plt

'''
%matplotlib inline

%load_ext autoreload
%autoreload 2
'''

from dataset import load_svhn
from knn import KNN
from metrics import binary_classification_metrics, multiclass_accuracy

'''

Загрузим и визуализируем данные
В задании уже дана функция load_svhn, загружающая данные с диска. Она возвращает данные для тренировки и для тестирования как numpy arrays.

Мы будем использовать цифры из датасета Street View House Numbers (SVHN, http://ufldl.stanford.edu/housenumbers/), чтобы решать задачу хоть сколько-нибудь сложнее MNIST.

'''

train_X, train_y, test_X, test_y = load_svhn("data", max_train=2000, max_test=200)

samples_per_class = 5  # Number of samples per class to visualize
plot_index = 1
for example_index in range(samples_per_class):
    for class_index in range(10):
        plt.subplot(5, 10, plot_index)
        image = train_X[train_y == class_index][example_index]
        plt.imshow(image.astype(np.uint8))
        plt.axis('off')
        plot_index += 1


'''

Сначала реализуем KNN для бинарной классификации
В качестве задачи бинарной классификации мы натренируем модель, которая будет отличать цифру 0 от цифры 9.

'''

# First, let's prepare the labels and the source data

# Only select 0s and 9s
binary_train_mask = (train_y == 0) | (train_y == 9)
binary_train_X = train_X[binary_train_mask]
binary_train_y = train_y[binary_train_mask] == 0

binary_test_mask = (test_y == 0) | (test_y == 9)
binary_test_X = test_X[binary_test_mask]
binary_test_y = test_y[binary_test_mask] == 0

# Reshape to 1-dimensional array [num_samples, 32*32*3]
binary_train_X = binary_train_X.reshape(binary_train_X.shape[0], -1)
binary_test_X = binary_test_X.reshape(binary_test_X.shape[0], -1)


# Create the classifier and call fit to train the model
# KNN just remembers all the data
knn_classifier = KNN(k=1)
knn_classifier.fit(binary_train_X, binary_train_y)

'''

Пришло время написать код!
Последовательно реализуйте функции compute_distances_two_loops, compute_distances_one_loop и compute_distances_no_loops в файле knn.py.

Эти функции строят массив расстояний между всеми векторами в тестовом наборе и в тренировочном наборе.
В результате они должны построить массив размера (num_test, num_train), где координата [i][j] соотвествует расстоянию между i-м вектором в test (test[i]) и j-м вектором в train (train[j]).

Обратите внимание Для простоты реализации мы будем использовать в качестве расстояния меру L1 (ее еще называют Manhattan distance).

'''

# TODO: implement compute_distances_two_loops in knn.py
dists = knn_classifier.compute_distances_two_loops(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

# TODO: implement compute_distances_one_loop in knn.py
dists = knn_classifier.compute_distances_one_loop(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

# TODO: implement compute_distances_no_loops in knn.py
dists = knn_classifier.compute_distances_no_loops(binary_test_X)
assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

'''
# Lets look at the performance difference
%timeit knn_classifier.compute_distances_two_loops(binary_test_X)
%timeit knn_classifier.compute_distances_one_loop(binary_test_X)
%timeit knn_classifier.compute_distances_no_loops(binary_test_X)
'''

# TODO: implement predict_labels_binary in knn.py
prediction = knn_classifier.predict(binary_test_X)

# TODO: implement binary_classification_metrics in metrics.py
precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print("KNN with k = %s" % knn_classifier.k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1)) 

# Let's put everything together and run KNN with k=3 and see how we do
knn_classifier_3 = KNN(k=3)
knn_classifier_3.fit(binary_train_X, binary_train_y)
prediction = knn_classifier_3.predict(binary_test_X)

precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print("KNN with k = %s" % knn_classifier_3.k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1)) 


'''

Кросс-валидация (cross-validation)
Попробуем найти лучшее значение параметра k для алгоритма KNN!

Для этого мы воспользуемся k-fold cross-validation (https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation). Мы разделим тренировочные данные на 5 фолдов (folds), и по очереди будем использовать каждый из них в качестве проверочных данных (validation data), а остальные -- в качестве тренировочных (training data).

В качестве финальной оценки эффективности k мы усредним значения F1 score на всех фолдах. После этого мы просто выберем значение k с лучшим значением метрики.

Бонус: есть ли другие варианты агрегировать F1 score по всем фолдам? Напишите плюсы и минусы в клетке ниже.

'''
'''
# Find the best k using cross-validation based on F1 score
num_folds = 5
train_folds_X = []
train_folds_y = []

# TODO: split the training data in 5 folds and store them in train_folds_X/train_folds_y
for i in range(num_folds):
    train_folds_X.append(binary_train_X[i*240:(i+1)*240])
    train_folds_y.append(binary_train_y[i*240:(i+1)*240])
k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
k_to_f1 = {}  # dict mapping k values to mean F1 scores (int -> float)

for k in k_choices: k_to_f1[k]=0

for i in range(num_folds):
    train_X_ = []
    train_y_ = []
    for j in range(num_folds):
        if i!=j:
            #print(*train_folds_X[j])
            train_X_.append(train_folds_X[j])
            train_y_.append(train_folds_y[i])
    train_X_ = np.concatenate(train_X_)
    train_y_ = np.concatenate(train_y_)
    for k in k_choices:
        knn_classifier = KNN(k)
        knn_classifier.fit(train_X_, train_y_)

        prediction = knn_classifier.predict(train_folds_X[i])
        
        precision, recall, f1, accuracy = binary_classification_metrics(prediction, train_folds_y[i])
        k_to_f1[k]+=f1

for k in sorted(k_to_f1):
    print('k = %d, f1 = %f' % (k, k_to_f1[k]/num_folds))
'''
'''

Проверим, как хорошо работает лучшее значение k на тестовых данных (test data)

'''

# TODO Set the best k to the best value found by cross-validation
'''
best_k = 8

best_knn_classifier = KNN(k=best_k)
best_knn_classifier.fit(binary_train_X, binary_train_y)
prediction = best_knn_classifier.predict(binary_test_X)

precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
print("Best KNN with k = %s" % best_k)
print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1)) 
'''
'''

Многоклассовая классификация (multi-class classification)
Переходим к следующему этапу - классификации на каждую цифру.

'''

# Now let's use all 10 classes
train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

knn_classifier = KNN(k=1)
knn_classifier.fit(train_X, train_y)

# TODO: Implement predict_labels_multiclass
predict = knn_classifier.predict(test_X)

# TODO: Implement multiclass_accuracy
accuracy = multiclass_accuracy(predict, test_y)
print("Accuracy: %4.2f" % accuracy)

'''

Снова кросс-валидация. Теперь нашей основной метрикой стала точность (accuracy), и ее мы тоже будем усреднять по всем фолдам.

'''
'''
num_folds = 5
train_folds_X = []
train_folds_y = []

# TODO: split the training data in 5 folds and store them in train_folds_X/train_folds_y
for i in range(num_folds):
    train_folds_X.append(train_X[i*400:(i+1)*400])
    train_folds_y.append(train_y[i*400:(i+1)*400])
k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
k_to_accuracy = {}  # dict mapping k values to mean F1 scores (int -> float)

for k in k_choices: k_to_accuracy[k]=0

for i in range(num_folds):
    train_X_ = []
    train_y_ = []
    for j in range(num_folds):
        if i!=j:
            #print(*train_folds_X[j])
            train_X_.append(train_folds_X[j])
            train_y_.append(train_folds_y[i])
    train_X_ = np.concatenate(train_X_)
    train_y_ = np.concatenate(train_y_)
    for k in k_choices:
        knn_classifier = KNN(k)
        knn_classifier.fit(train_X_, train_y_)

        prediction = knn_classifier.predict(train_folds_X[i])
        
        accuracy = multiclass_accuracy(prediction, train_folds_y[i])
        k_to_accuracy[k]+=accuracy

for k in sorted(k_to_accuracy):
    print('k = %d, f1 = %f' % (k, k_to_accuracy[k]/num_folds))
'''
'''

Финальный тест - классификация на 10 классов на тестовой выборке (test data)
Если все реализовано правильно, вы должны увидеть точность не менее 0.2.

'''

# TODO Set the best k as a best from computed
best_k = 50

best_knn_classifier = KNN(k=best_k)
best_knn_classifier.fit(train_X, train_y)
prediction = best_knn_classifier.predict(test_X)

# Accuracy should be around 20%!
accuracy = multiclass_accuracy(prediction, test_y)
print("Accuracy: %4.2f" % accuracy)
