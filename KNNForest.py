import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import random as rn
eps = np.finfo(float).eps




class DecisionTreeNode:
    def __init__(self, feature_clasificator, children, default_classifier):
        self.feature_clasificator = feature_clasificator
        self.children = children
        self.default_classifier = default_classifier

    def predict(self, example):
        if len(self.children) == 0:
            return self.default_classifier
        if self.feature_clasificator.classify_example(example):
            return self.children[0].predict(example)
        else:
            return self.children[1].predict(example)


class feature:
    def __init__(self, feature_id, threshold):
        self.feature_id = feature_id
        self.threshold = threshold

    def classify_example(self, example):
        if example[self.feature_id] >= self.threshold:
            return True
        else:
            return False


def entropy(examples):
    # leaf case
    if len(examples) == 0:
        return 0

    # otherwise
    prob_healthy_example = len([person for person in examples if person[0] == 'B']) / len(examples)
    prob_unhealthy_example = len([person for person in examples if person[0] == 'M']) / len(examples)
    if prob_healthy_example == 0 or prob_unhealthy_example == 0:
        return 0

    return -1 * (prob_healthy_example * np.log2(prob_healthy_example) +
                 prob_unhealthy_example * np.log2(prob_unhealthy_example))


def IG(f, examples):
    subTree1 = []
    subTree2 = []

    for example in examples:
        if f.classify_example(example):
            subTree1.append(example)
        else:
            subTree2.append(example)

    subTree1_value = (len(subTree1) / len(examples)) * entropy(subTree1)
    subTree2_value = (len(subTree2) / len(examples)) * entropy(subTree2)

    return entropy(examples) - subTree1_value - subTree2_value


def MaxIG(features, examples):
    max_gain = -float('inf')
    feature_to_use = None

    for f_id in features:
        f_values = []
        for example in examples:
            f_values.append(example[f_id])

        f_values = np.unique(f_values)
        f_values.sort()  # sort the column of the specific feature

        if len(f_values) == 1:
            continue

        tmp_best_gain = -float('inf')
        best_threshold_for_curr_f = None

        for i in range(len(f_values) - 1):
            threshold = (f_values[i] + f_values[i + 1]) / 2
            new_feature = feature(f_id, threshold)
            ig_f = IG(new_feature, examples)
            if ig_f > tmp_best_gain:
                tmp_best_gain = ig_f
                best_threshold_for_curr_f = new_feature

        if tmp_best_gain >= max_gain:
            max_gain = tmp_best_gain
            feature_to_use = best_threshold_for_curr_f

    return feature_to_use


def MajorityClass(examples):
    number_of_healthy_people = len([person for person in examples if person[0] == 'B'])
    if number_of_healthy_people < len(examples) / 2:
        return 'M'
    else:
        return 'B'


# default classification is majority classification
# SelectFeature = IG Function
# decision_noe is class:
# first argument is the feature it will classify by
# second argument is the children (two in our case)
# third argument is the default_classification A.K.A majority in current state
# MEPN - MINIMAL EXAMPLES PER NODE
def TDIDT(examples, features, default_classification, SelectFeature, MEPN):
    if len(examples) == 0 or len(examples) < MEPN:
        return DecisionTreeNode(None, [], default_classification)

    c = MajorityClass(examples)

    # check if consistent node
    if len([person for person in examples if person[0] == 'B']) == 0 or \
            len([person for person in examples if person[0] == 'M']) == 0 or len(features) == 0:
        return DecisionTreeNode(None, [], c)

    f = SelectFeature(features, examples)   # here we get a feature class object which has the feature id and threshold


    subTree1_examples = []
    subTree2_examples = []

    for example in examples:
        if f.classify_example(example):
            subTree1_examples.append(example)
        else:
            subTree2_examples.append(example)

    subTree1 = TDIDT(subTree1_examples, features, c, SelectFeature, MEPN)
    subTree2 = TDIDT(subTree2_examples, features, c, SelectFeature, MEPN)

    return DecisionTreeNode(f, [subTree1, subTree2], c)


# MEPN - MINIMAL EXAMPLES PER NODE
def ID3(examples, features, MEPN):
    c = MajorityClass(examples)
    return TDIDT(examples, features, c, MaxIG, MEPN)


class ID3_algorythm:
    def __init__(self, MEPN):
        self.ID3_root = None
        self.MEPN = MEPN

    def fit(self, examples, features):
        self.ID3_root = ID3(examples, features, self.MEPN)

    def predict(self, test_obj):
        return self.ID3_root.predict(test_obj)


# def id3_general_run(MEPN):
#     data = pd.read_csv("train.csv")
#     examples = np.array(data)
#     features = [i for i in range(1, 31)]
#
#     id3 = ID3_algorythm(MEPN)
#     id3.fit(examples, features)
#
#     tests = np.array(pd.read_csv("test.csv"))
#
#     cnt = 0
#     for test in tests:
#         if id3.predict(test) == test[0]:
#             cnt += 1
#
#     print(cnt / len(tests))


# def loss_calculation():
#     data = pd.read_csv("train.csv")
#     examples = np.array(data)
#     features = [i for i in range(1, 31)]
#
#     id3 = ID3_algorythm(1)
#     id3.fit(examples, features)
#
#     tests = np.array(pd.read_csv("test.csv"))
#
#     FP = 0
#     FN = 0
#     for test in tests:
#         result = id3.predict(test)
#         if result != test[0] and result == 'B':
#             FN += 1
#         if result != test[0] and result == 'M':
#             FP += 1
#
#     loss = (0.1 * FP + FN) / len(tests)
#     print(loss)

# def run_with_min_avg_max_results(p, N, K, MEPN):
#     min_result = 1
#     max_result = 0
#     results_list = []
#     for i in range(0, 20):
#         curr_result = test_with_params(p, N, K, MEPN)
#         if curr_result > max_result:
#             max_result = curr_result
#         if curr_result < min_result:
#             min_result = curr_result
#         results_list.append(curr_result)
#     print("min accuracy ==> ", min_result)
#     print("max accuracy ==> ", max_result)
#     sum = 0
#     for result in results_list:
#         sum += result
#     print("avg accuracy ==> ", sum / len(results_list))


def main():

    # params [0.5, 60, 15, 1] / [0.4, 8, 7, 1]
    print(test_with_params(0.4, 8, 7, 1))

    # uncomment to runs the experiment to find the params best suit for the algorythm
    # params = experiment()
    # print(params)

    # uncomment to test with 20 runs and calculate min max and avg accuracy
    # run_with_min_avg_max_results(0.4, 8, 7, 1)


# uncomment to to run the experiment for params accuracy
# def experiment():
#     M = [1]     # , 2, 3, 5, 8, 16, 30, 50, 80, 120]
#     N = [2, 5, 8, 10, 15, 20, 40, 60, 90, 100]
#     K = [2, 4, 5, 7, 10, 15, 25, 30, 50, 70]
#     P = [0.3, 0.32, 0.36, 0.4, 0.5, 0.6, 0.7]
#
#     data = pd.read_csv("train.csv")
#     all_examples = np.array(data)
#     all_features = [i for i in range(1, 31)]
#
#     best_result = 0
#     top_params = []
#
#     accuracy = []   # add to this list for graph purposes
#     for m in M:
#         for n in N:
#             for k in K:
#                 if k >= n:
#                     continue
#                 for prob in P:
#                     results_per_params = []
#                     kf = KFold(n_splits=5, shuffle=True, random_state=207079922)
#                     for train_index, test_index in kf.split(all_examples):
#
#                         # prepare examples in a list
#                         tmp_all_examples = all_examples[train_index]
#
#                         # KNN
#                         n_len = len(tmp_all_examples)
#                         N_trees = []
#                         N_centroids = []
#
#                         for i in range(1, n + 1):
#                             # prepare examples for curr tree
#                             tmp_examples_for_train = []
#                             tmp_example_indexes = rn.sample(range(0, n_len), int(n_len * prob))
#                             for tmp_example in tmp_example_indexes:
#                                 tmp_examples_for_train.append(tmp_all_examples[tmp_example])
#
#                             # calculate centroid for curr tree
#                             curr_tree_centroid = calcCentroid(tmp_examples_for_train, all_features)
#                             N_centroids.append(curr_tree_centroid)
#
#                             # train tree
#                             tmp_id3 = ID3_algorythm(m)
#                             tmp_id3.fit(tmp_examples_for_train, all_features)
#                             N_trees.append(tmp_id3)
#
#
#                         tmp_all_tests = all_examples[test_index]
#
#                         results = []
#
#                         for test in tmp_all_tests:
#                             tmp_test = test[1:]
#                             dist = []
#                             # calc distance from centroids
#                             for centroid in N_centroids:
#                                 dist.append(np.linalg.norm(tmp_test - centroid))
#
#                             # sort the distances
#                             res = {i: dist[i] for i in range(len(dist))}
#                             sorted_res = dict(sorted(res.items(), key=lambda item: item[1]))
#
#                             Pos = 0
#                             Neg = 0
#                             # choose nearest K
#                             for i in range(k):
#                                 curr_id = list(sorted_res.items())[i][0]
#                                 result = N_trees[curr_id].predict(test)
#                                 if result == 'M':
#                                     Neg += 1
#                                 else:
#                                     Pos += 1
#
#                             if Neg >= Pos:
#                                 results.append('M')
#                             else:
#                                 results.append('B')
#
#                         cnt = 0
#                         i = 0
#                         for test in tmp_all_tests:
#                             if results[i] == test[0]:
#                                 cnt += 1
#                             i += 1
#
#                         tempo_result = cnt / len(tmp_all_tests)
#                         results_per_params.append(tempo_result)
#
#                     avg_result = sum(results_per_params[i] for i in range(len(results_per_params)))/len(results_per_params)
#
#                     print("avg result ==> ", avg_result, "params ==> ", [n, k, prob], " <==")
#                     if avg_result > best_result:
#                         best_result = avg_result
#                         if avg_result > 0.98:
#                             top_params.append([n, k, prob])
#
#     print(top_params)
#     return best_result


def calcCentroid(examples_subset_ids, all_features):

    centroid = []
    for F in all_features:
        centroid_sum = 0
        for example in examples_subset_ids:
            centroid_sum += example[F]
        avg = centroid_sum / len(examples_subset_ids)
        centroid.append(avg)

    return centroid


def test_with_params(p, N, K, MEPN):
    data = pd.read_csv("train.csv")
    all_examples = np.array(data)
    all_features = [i for i in range(1, 31)]
    n = len(all_examples)

    N_trees = []
    N_centroids = []

    for i in range(1, N+1):
        # prepare examples for curr tree
        tmp_examples_for_train = []
        tmp_example_indexes = rn.sample(range(0, n), int(n * p))
        for tmp_example in tmp_example_indexes:
            tmp_examples_for_train.append(all_examples[tmp_example])

        # calculate centroid for curr tree
        curr_tree_centroid = calcCentroid(tmp_examples_for_train, all_features)
        N_centroids.append(curr_tree_centroid)

        # train tree
        tmp_id3 = ID3_algorythm(MEPN)
        tmp_id3.fit(tmp_examples_for_train, all_features)
        N_trees.append(tmp_id3)


    tests = np.array(pd.read_csv("test.csv"))



    results = []

    for test in tests:
        tmp_test = test[1:]
        dist = []
        # calc distance from centroids
        for centroid in N_centroids:
            dist.append(np.linalg.norm(tmp_test-centroid))

        # sort the distances
        res = {i: dist[i] for i in range(len(dist))}
        sorted_res = dict(sorted(res.items(), key=lambda item: item[1]))


        P = 0
        N = 0
        # choose nearest K
        for i in range(K):
            curr_id = list(sorted_res.items())[i][0]
            result = N_trees[curr_id].predict(test)
            if result == 'M':
                N += 1
            else:
                P += 1

        if N >= P:
            results.append('M')
        else:
            results.append('B')



    cnt = 0
    i = 0
    for test in tests:
        if results[i] == test[0]:
            cnt += 1
        i += 1

    return cnt / len(tests)



if __name__ == '__main__':
    main()
