import numpy as np
from collections import Counter


class Node:
    def __init__(self, left=None, right=None, predicate=None, value=None):
        if not predicate and not value:
            raise Exception("node should contain predicate or value")
        self.left = left
        self.right = right
        self.predicate = predicate
        self.value = value


def gain(index_fun):
    def fun(x, y, idx, predicate):
        false_idx, true_idx = [], []
        pfalse, ptrue = 0, 0
        psample = 1.0 / len(idx)
        for i in idx:
            if predicate(x[i]):
                true_idx.append(i)
                ptrue += psample
            else:
                false_idx.append(i)
                pfalse += psample
        return index_fun(y, idx) - pfalse * index_fun(y, false_idx) - ptrue * index_fun(y, true_idx)
    return fun


def entropy(y, idx):
    n = len(idx)
    c = Counter()
    for i in idx:
        c[y[i]] += 1
    res = 0
    for key in dict(c):
        prob = c[key] / n
        res -= prob * np.log2(prob)
    return res


def gini(y, idx):
    n = len(idx)
    c = Counter()
    for i in idx:
        c[y[i]] += 1
    res = 1
    for key in dict(c):
        prob = c[key] / n
        res -= prob * prob
    return res


class DecisionTree:
    def __init__(self, criterion):
        self.root = None

        if criterion == 'entropy':
            self.criterion = gain(entropy)
        elif criterion == 'gini':
            self.criterion = gain(gini)
        else:
            raise Exception('not supported criterion')

    def fit(self, x, y, feature_ids):
        def create_tree(train_idx):
            best_value, best_predicate = -1, lambda x: False
            for feature_id in feature_ids:
                sorted_idx = sorted(train_idx, key=lambda i: x[i][feature_id])
                for cur_id, next_id in zip(sorted_idx, sorted_idx[1:]):
                    x_cur, x_next = x[cur_id][feature_id], x[next_id][feature_id]
                    if x_cur == x_next:
                        continue

                    def create_predicate(val, feature):
                        def pred(xx):
                            return xx[feature] > val
                        setattr(pred, 'feature', feature)
                        setattr(pred, 'value', val)
                        return pred

                    predicate = create_predicate((x_cur + x_next) / 2, feature_id)
                    value = self.criterion(x, y, train_idx, predicate)
                    if value > best_value:
                        best_value, best_predicate = value, predicate
            if best_value < 1e-9:
                left_idx, right_idx = np.array([]), train_idx
            else:
                left_idx = np.array([i for i in train_idx if not best_predicate(x[i])])
                right_idx = np.array([i for i in train_idx if best_predicate(x[i])])

            def most_frequent_class(idx):
                c = Counter()
                for i in idx:
                    c[y[i]] += 1
                return c.most_common()[0][0]

            if left_idx.size == 0:
                return Node(value=most_frequent_class(right_idx))
            if right_idx.size == 0:
                return Node(value=most_frequent_class(left_idx))
            return Node(left=create_tree(left_idx), right=create_tree(right_idx), predicate=best_predicate)

        train_idx = np.arange(len(x))
        self.root = create_tree(train_idx)

    def predict(self, x):
        def get_value(node, sample):
            if node.value:
                return node.value
            if node.predicate(sample):
                return get_value(node.right, sample)
            else:
                return get_value(node.left, sample)

        return [get_value(self.root, sample) for sample in x]

    def print(self):
        def print_node(node, indent):
            if node.value:
                print(indent, node.value)
            else:
                print(indent, getattr(node.predicate, 'feature'), getattr(node.predicate, 'value'))
                print_node(node.left, indent + "  ")
                print_node(node.right, indent + "  ")
        print_node(self.root, "")


class RandomForest:
    def __init__(self, n_estimators, criterion):
        self.criterion = criterion
        self.trees = [DecisionTree(criterion) for _ in range(n_estimators)]
        self.n_samples = 0
        self.n_features = 0
        self.feature_importances_ = None

    def print(self):
        for id, tree in enumerate(self.trees):
            print(id, ':')
            tree.print()

    def fit(self, x, y):
        self.n_samples, self.n_features = x.shape
        self.feature_importances_ = np.zeros(self.n_features)
        features_in_tree = np.int(np.sqrt(self.n_features))
        trees_sample_ids = []
        for tree in self.trees:
            idx = np.random.choice(self.n_samples, size=self.n_samples)
            trees_sample_ids.append(idx)
            tree_x, tree_y = x[idx], y[idx]
            print('tx', tree_x)
            print('ty', tree_y)
            feature_ids = np.random.choice(self.n_features, size=features_in_tree, replace=False)
            tree.fit(tree_x, tree_y, feature_ids)
        self.calc_feature_importances(x, y, trees_sample_ids)
        print('features', self.feature_importances_)

    def calc_feature_importances(self, x, y, trees_sample_ids):
        def calc_out_of_bag_error(id):
            error = 0
            count_trees_without_id = 0
            for tree, tree_ids in zip(self.trees, trees_sample_ids):
                if not id in tree_ids:
                    count_trees_without_id += 1
                    y_pred = tree.predict([x[id]])[0]
                    if y_pred != y[id]:
                        error += 1
            if count_trees_without_id == 0:
                return 0
            return error / count_trees_without_id

        original_errors = [calc_out_of_bag_error(id) for id in range(self.n_samples)]
        for feature in range(self.n_features):
            save_feature_column = np.copy(x[:, feature])
            np.random.shuffle(x[:, feature])
            errors_diff = [calc_out_of_bag_error(id) - original_errors[id] for id in range(self.n_samples)]
            if errors_diff.count(errors_diff[0]) == self.n_samples:  # all same value
                self.feature_importances_[feature] = 0
            else:
                mean = np.mean(errors_diff)
                std = np.std(errors_diff)
                self.feature_importances_[feature] = mean / std
            x[:, feature] = save_feature_column

    def predict(self, x):
        counters = [Counter() for _ in range(x.shape[0])]
        for tree in self.trees:
            for y, c in zip(tree.predict(x), counters):
                c[y] += 1
        y = [c.most_common()[0][0] for c in counters]
        return y
