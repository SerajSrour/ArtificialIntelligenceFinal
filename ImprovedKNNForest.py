import numpy as np
import pandas as pd
# from sklearn.model_selection import KFold
# import matplotlib.pyplot as plt
import random as rn
# from sklearn import preprocessing
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

    weights = [0, 0.0004201680672268907, 0.0008220679576178298, 0.0008281573498964817, 6.938893903907229e-19, 7.930164461608261e-19,
               0.00041407867494823957, -6.089392278650196e-06, 0.0037327974668128127, -0.0012544148094020215, -0.0012665935939593232,
               9.912705577010326e-20, 0.0008342467421751307, -1.9825411154020652e-19, 0.00041407867494824027, 0.00040798928266958846,
               0.0008342467421751318, -0.0004201680672268911, -7.930164461608261e-19, -8.921435019309293e-19, -1.7347234759768072e-19,
               0.000840336134453782, -0.009499451954694921, -0.002874193155523078, 0.0016563146997929609, 0.0016684934843502626,
               -0.002904640116916332, -0.0029472658628668857, 0.002447935696017538, 0.004146876141761052, 0.00462793813177445]


    for i in range(len(weights)):
        weights[i] = weights[i]+1

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
            ig_f = IG(new_feature, examples) * weights[f_id]
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

    f = SelectFeature(features, examples)  # here we get a feature class object which has the feature id and threshold

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
    print(test_with_params(0.4, 8, 7, 3))

    # uncomment to runs the experiment to find the params best suit for the algorythm
    # params = experiment()
    # print(params)

    # uncomment to test with 20 runs and calculate min max and avg accuracy
    # this function calculate accuracy on test file
    # run_with_min_avg_max_results(0.4, 8, 7, 3)

    # uncomment to enable second improvment only
    # improved_KNN(0.4, 8, 7, 3)

    # uncomment to enable weight calculation
    # improved_K_fold_experiment(0.4, 8, 7, 1)

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
#                     # avg_result for current params
#                     avg_result = sum(results_per_params[i] for i in range(len(results_per_params)))/len(results_per_params)
#
#                     print("avg result ==> ", avg_result, "params ==> ", [n, k, prob], " <==")
#                     # if avg_result > best_result:
#                     #     best_result = avg_result
#                     #     if avg_result > 0.98:
#                     #         top_params.append([n, k, prob])
#
#     # print(top_params)
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
    # scaler = preprocessing.MinMaxScaler()
    # data_columns = data[data.columns[1:]]
    # d = scaler.fit_transform(data_columns)
    # data[data.columns[1:]] = d

    all_examples = np.array(data)
    all_features = [i for i in range(1, 31)]
    n = len(all_examples)

    N_trees = []
    N_centroids = []

    for i in range(1, N + 1):
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

    tests_data = pd.read_csv("test.csv")
    # scaler = preprocessing.MinMaxScaler()
    # data_columns = tests_data[tests_data.columns[1:]]
    # d = scaler.fit_transform(data_columns)
    # tests_data[data.columns[1:]] = d
    tests = np.array(tests_data)

    results = []

    for test in tests:
        tmp_test = test[1:]
        dist = []
        # calc distance from centroids
        for centroid in N_centroids:
            dist.append(np.linalg.norm(tmp_test - centroid))

        # sort the distances
        res = {i: dist[i] for i in range(len(dist))}
        sorted_res = dict(sorted(res.items(), key=lambda item: item[1]))

        # calculate avg distance for positive trees and negative trees.
        pos_trees_distances = []
        neg_trees_distances = []

        Pos = 0
        Neg = 0
        # choose nearest K
        for i in range(K):
            curr_id = list(sorted_res.items())[i][0]
            result = N_trees[curr_id].predict(test)
            if result == 'M':
                Neg += 1
                neg_trees_distances.append(list(sorted_res.items())[i][1])
            else:
                Pos += 1
                pos_trees_distances.append(list(sorted_res.items())[i][1])

        if Neg >= 1.2 * Pos:
            results.append('M')
        elif Pos >= 1.2 * Neg:
            results.append('B')
        else:
            pos_avg_dist = sum(pos_trees_distances[i] for i in range(len(pos_trees_distances))) / len(
                pos_trees_distances)
            neg_avg_dist = sum(neg_trees_distances[i] for i in range(len(neg_trees_distances))) / len(
                neg_trees_distances)
            if neg_avg_dist >= pos_avg_dist:
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


# # improvement trial 1
# def improved_KNN(p, N, K, MEPN):
#     # read data
#     data = pd.read_csv("train.csv")
#     scaler = preprocessing.MinMaxScaler()
#     data_columns = data[data.columns[1:]]
#     d = scaler.fit_transform(data_columns)
#     data[data.columns[1:]] = d
#     all_examples = np.array(data)
#     all_features = [i for i in range(1, 31)]
#
#     kfold_tests_results = []
#     kf = KFold(n_splits=5, shuffle=True, random_state=207079922)
#     # iterate over a number of tests
#     for train_index, test_index in kf.split(all_examples):
#         # prepare examples in a list
#         tmp_all_examples = all_examples[train_index]
#
#         # KNN
#         n_len = len(tmp_all_examples)
#         N_trees = []
#         N_centroids = []
#
#         for i in range(1, N + 1):
#             # prepare examples for curr tree
#             # tmp_examples_for_train is a list for randomly chosen examples
#             tmp_examples_for_train = []
#             tmp_example_indexes = rn.sample(range(0, n_len), int(n_len * p))
#             for tmp_example in tmp_example_indexes:
#                 tmp_examples_for_train.append(tmp_all_examples[tmp_example])
#
#             # calculate centroid for curr tree
#             curr_tree_centroid = calcCentroid(tmp_examples_for_train, all_features)
#             N_centroids.append(curr_tree_centroid)
#
#             # train tree
#             tmp_id3 = ID3_algorythm(MEPN)
#             tmp_id3.fit(tmp_examples_for_train, all_features)
#             N_trees.append(tmp_id3)
#
#
#         # prepare the test examples in a list for the current test
#         tmp_all_tests = all_examples[test_index]
#         results = []
#
#         # iterate over the test examples in the current test
#         for test in tmp_all_tests:
#             tmp_test = test[1:]
#             dist = []
#             # calc distance from centroids
#             for centroid in N_centroids:
#                 dist.append(np.linalg.norm(tmp_test - centroid))
#
#             # sort the distances
#             res = {i: dist[i] for i in range(len(dist))}
#             sorted_res = dict(sorted(res.items(), key=lambda item: item[1]))
#             # calculate avg distance for positive trees and negative trees.
#             pos_trees_distances = []
#             neg_trees_distances = []
#
#             Pos = 0
#             Neg = 0
#             # choose nearest K
#             for i in range(K):
#                 curr_id = list(sorted_res.items())[i][0]
#                 result = N_trees[curr_id].predict(test)
#                 if result == 'M':
#                     Neg += 1
#                     neg_trees_distances.append(list(sorted_res.items())[i][1])
#                 else:
#                     Pos += 1
#                     pos_trees_distances.append(list(sorted_res.items())[i][1])
#
#             if Neg >= 1.2 * Pos:
#                 results.append('M')
#             elif Pos >= 1.2 * Neg:
#                 results.append('B')
#             else:
#                 pos_avg_dist = sum(pos_trees_distances[i] for i in range(len(pos_trees_distances))) / len(
#                     pos_trees_distances)
#                 neg_avg_dist = sum(neg_trees_distances[i] for i in range(len(neg_trees_distances))) / len(
#                     neg_trees_distances)
#                 if neg_avg_dist >= pos_avg_dist:
#                     results.append('M')
#                 else:
#                     results.append('B')
#
#         # calc result for the current test
#         cnt = 0
#         i = 0
#         for test in tmp_all_tests:
#             if results[i] == test[0]:
#                 cnt += 1
#             i += 1
#
#         # add results to the list
#         tempo_result = cnt / len(tmp_all_tests)
#         kfold_tests_results.append(tempo_result)
#
#
#
#     avg_result = sum(kfold_tests_results[i] for i in range(len(kfold_tests_results))) / len(kfold_tests_results)
#     print("avg result ==> ", avg_result)


# def calc_error_in_tests(tmp_all_tests, N_trees, N_centroids, K):
#     # results contain our ML algorythm predictions
#     # results = []
#
#     # iterate over the test examples in the current test
#     K_trees_results = []
#     for tre in range(K):
#         K_trees_results.append(0)
#
#     for test in tmp_all_tests:
#         tmp_test = test[1:]
#         dist = []
#         # calc distance from centroids
#         for centroid in N_centroids:
#             dist.append(np.linalg.norm(tmp_test - centroid))
#
#         # sort the distances
#         res = {i: dist[i] for i in range(len(dist))}
#         sorted_res = dict(sorted(res.items(), key=lambda item: item[1]))
#
#         # calculate avg distance for positive trees and negative trees.
#         # pos_trees_distances = []
#         # neg_trees_distances = []
#
#         for i in range(K):
#             curr_id = list(sorted_res.items())[i][0]
#             result = N_trees[curr_id].predict(test)
#
#             if result == 'M' and test[0] != result:
#                 K_trees_results[i] += 1
#             elif result == 'B' and test[0] != result:
#                 K_trees_results[i] += 1
#
#
#     for j in range(K):
#         K_trees_results[j] = K_trees_results[j]/len(tmp_all_tests)
#
#     return K_trees_results
#
#
# def run_with_without_feature_i(data, all_examples, all_features, p, N, K, MEPN):
#     new_features_set = [i for i in range(1, 30)]
#     weights_to_return = []
#
#     kf = KFold(n_splits=5, shuffle=True, random_state=207079922)
#     # iterate over a number of tests
#     for train_index, test_index in kf.split(all_examples):
#         # features_weights in 5 test in kfold
#         weights = []
#
#         for specific_feature in all_features:
#             print("----> " ,specific_feature)
#             # remove specific feature from features list
#             # prepare examples without specific feature
#             data_without_f = pd.DataFrame.copy(data)
#             del data_without_f[data_without_f.columns[specific_feature]]
#             all_examples_without_f = np.array(data_without_f)
#
#
#             # prepare examples in a list
#             tmp_all_examples = all_examples[train_index]
#             tmp_all_examples_without_f = all_examples_without_f[train_index]
#
#             # KNN
#             n_len = len(tmp_all_examples)
#             N_trees = []
#             N_centroids = []
#
#             N_trees_without_feature = []
#             N_centroids_without_features = []
#
#             for i in range(1, N + 1):
#                 # prepare examples for curr tree
#                 # tmp_examples_for_train is a list for randomly chosen examples
#                 # tmp_examples_for_train_without_f is simply as above in data without specific feature
#                 tmp_examples_for_train = []
#                 tmp_examples_for_train_without_f = []
#
#                 # sample the indexes for train
#                 tmp_example_indexes = rn.sample(range(0, n_len), int(n_len * p))
#                 for tmp_example in tmp_example_indexes:
#                     tmp_examples_for_train.append(tmp_all_examples[tmp_example])
#                     tmp_examples_for_train_without_f.append(tmp_all_examples_without_f[tmp_example])
#
#
#                 # calculate centroid for curr tree
#                 curr_tree_centroid = calcCentroid(tmp_examples_for_train, all_features)
#                 N_centroids.append(curr_tree_centroid)
#                 N_centroids_without_features.append(calcCentroid(tmp_examples_for_train_without_f, new_features_set))
#
#                 # train tree
#                 tmp_id3 = ID3_algorythm(MEPN)
#                 tmp_id3.fit(tmp_examples_for_train, all_features)
#                 N_trees.append(tmp_id3)
#
#                 # train tree without feature
#                 tmp_id3_without_f = ID3_algorythm(MEPN)
#                 tmp_id3_without_f.fit(tmp_examples_for_train_without_f, new_features_set)
#                 N_trees_without_feature.append(tmp_id3_without_f)
#
#
#             # -------------------------------------------------
#             # ----------------- TEST TIME ---------------------
#             # -------------------------------------------------
#
#             # prepare the test examples in a list for the current test
#             tmp_all_tests = all_examples[test_index]
#             tmp_all_tests_without_f = all_examples_without_f[test_index]
#
#             error_with_feature_K_trees = calc_error_in_tests(tmp_all_tests, N_trees, N_centroids, K)
#             error_without_feature_K_trees = calc_error_in_tests(tmp_all_tests_without_f, N_trees_without_feature, N_centroids_without_features, K)
#
#             # print("1   ", error_with_feature_K_trees)
#             # print("2   ", error_without_feature_K_trees)
#             sum_of_all_trees_error = 0
#             for h in range(K):
#                 sum_of_all_trees_error += (error_with_feature_K_trees[h] - error_without_feature_K_trees[h])
#
#             final_avg_error = sum_of_all_trees_error / K
#             weights.append(final_avg_error)
#
#
#         weights_to_return.append(weights)
#         print(weights_to_return)
#
#     return weights_to_return
#
#
# def improved_K_fold_experiment(p, N, K, MEPN):
#     # read data
#     data = pd.read_csv("train.csv")
#     scaler = preprocessing.MinMaxScaler()
#     data_columns = data[data.columns[1:]]
#     d = scaler.fit_transform(data_columns)
#     data[data.columns[1:]] = d
#     all_examples = np.array(data)
#     all_features = [i for i in range(1, 31)]
#
#     weight_list = run_with_without_feature_i(data, all_examples, all_features, p, N, K, MEPN)
#
#     weight_avg_list = []
#     for i in range(len(weight_list[0])):
#         sum_weights = 0
#         for j in range(len(weight_list)):
#             sum_weights += weight_list[j][i]
#         weight_avg_list.append(sum_weights / len(weight_list))
#
#     print(weight_avg_list)


if __name__ == '__main__':
    main()
