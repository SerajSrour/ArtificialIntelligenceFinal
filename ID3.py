import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
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


def id3_general_run(MEPN):
    data = pd.read_csv("train.csv")
    examples = np.array(data)
    features = [i for i in range(1, 31)]

    id3 = ID3_algorythm(MEPN)
    id3.fit(examples, features)

    tests = np.array(pd.read_csv("test.csv"))

    cnt = 0
    for test in tests:
        if id3.predict(test) == test[0]:
            cnt += 1

    print(cnt / len(tests))


def loss_calculation():
    data = pd.read_csv("train.csv")
    examples = np.array(data)
    features = [i for i in range(1, 31)]

    id3 = ID3_algorythm(1)
    id3.fit(examples, features)

    tests = np.array(pd.read_csv("test.csv"))

    FP = 0
    FN = 0
    for test in tests:
        result = id3.predict(test)
        if result != test[0] and result == 'B':
            FN += 1
        if result != test[0] and result == 'M':
            FP += 1

    loss = (0.1 * FP + FN) / len(tests)
    print(loss)


def main():
    # to enable and run question 1 uncomment the follow line
    id3_general_run(0)

    # to enable and run questions 3.3 uncomment the follow line
    # experiment()

    # to enable and run questions 3.4 uncomment the follow line
    # id3_general_run(1)

    # to enable and run questions 3.3 uncomment the follow line
    # loss_calculation()



# uncomment to enable experiment
# def experiment():
#     data = pd.read_csv("train.csv")
#
#     all_examples = np.array(data)
#     all_features = [i for i in range(1, 31)]
#
#
#     M = [1, 2, 3, 5, 8, 16, 30, 50, 80, 120]
#
#     accuracy = []
#     for i in M:
#         results = []
#         kf = KFold(n_splits=5, shuffle=True, random_state=207079922)
#         for train_index, test_index in kf.split(all_examples):
#             id3 = ID3_algorythm(i)
#             id3.fit(all_examples[train_index], all_features)
#             internal_cnt = 0
#             for test in all_examples[test_index]:
#                 if id3.predict(test) == test[0]:
#                     internal_cnt += 1
#             tmp_result = internal_cnt / len(all_examples[test_index])
#             results.append(tmp_result)
#
#         sum_results = 0
#         for result in results:
#             sum_results += result
#
#         accuracy.append(sum_results / len(results))
#
#     # to print the graph undo the next lines starting with plt
#     print(accuracy)
#     plt.plot(M, accuracy)
#     plt.ylabel('Solution Accuracy')
#     plt.xlabel('M - Minimal number of examples per node')
#     plt.title('M effect on solution accuracy')
#     plt.grid(True)
#     plt.show()



if __name__ == '__main__':
    main()
