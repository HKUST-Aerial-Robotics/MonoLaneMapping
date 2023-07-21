'''
reference: https://www.topcoder.com/community/competitive-programming/tutorials/assignment-problem-and-hungarian-algorithm/
'''

import numpy as np

def get_km_match(weights):
    matcher = KMMatcher(weights)
    return matcher.solve()

#max weight assignment
class KMMatcher:

    ## weights : nxm weight matrix (numpy , float), n <= m
    def __init__(self, weights):
        weights = np.array(weights).astype(np.float32)
        self.weights = weights
        self.n, self.m = weights.shape
        assert self.n <= self.m
        # init label
        self.label_x = np.max(weights, axis=1)
        self.label_y = np.zeros((self.m, ), dtype=np.float32)

        self.max_match = 0
        self.xy = -np.ones((self.n,), dtype=np.int32)
        self.yx = -np.ones((self.m,), dtype=np.int32)

    def do_augment(self, x, y):
        self.max_match += 1
        while x != -2:
            self.yx[y] = x
            ty = self.xy[x]
            self.xy[x] = y
            x, y = self.prev[x], ty

    def find_augment_path(self):
        self.S = np.zeros((self.n,), np.bool_)
        self.T = np.zeros((self.m,), np.bool_)

        self.slack = np.zeros((self.m,), dtype=np.float32)
        self.slackyx = -np.ones((self.m,), dtype=np.int32)  # l[slackyx[y]] + l[y] - w[slackx[y], y] == slack[y]

        self.prev = -np.ones((self.n,), np.int32)

        queue, st = [], 0
        root = -1

        for x in range(self.n):
            if self.xy[x] == -1:
                queue.append(x);
                root = x
                self.prev[x] = -2
                self.S[x] = True
                break

        self.slack = self.label_y + self.label_x[root] - self.weights[root]
        self.slackyx[:] = root

        while True:
            while st < len(queue):
                x = queue[st]; st+= 1

                is_in_graph = np.isclose(self.weights[x], self.label_x[x] + self.label_y)
                nonzero_inds = np.nonzero(np.logical_and(is_in_graph, np.logical_not(self.T)))[0]

                for y in nonzero_inds:
                    if self.yx[y] == -1:
                        return x, y
                    self.T[y] = True
                    queue.append(self.yx[y])
                    self.add_to_tree(self.yx[y], x)

            self.update_labels()
            queue, st = [], 0
            is_in_graph = np.isclose(self.slack, 0)
            nonzero_inds = np.nonzero(np.logical_and(is_in_graph, np.logical_not(self.T)))[0]

            for y in nonzero_inds:
                x = self.slackyx[y]
                if self.yx[y] == -1:
                    return x, y
                self.T[y] = True
                if not self.S[self.yx[y]]:
                    queue.append(x)
                    self.add_to_tree(self.yx[y], x)

    def solve(self):
        while self.max_match < self.n:
            x, y = self.find_augment_path()
            self.do_augment(x, y)

        A = []
        for x in range(self.n):
            if self.weights[x, self.xy[x]] == 0:
                continue
            A.append([x, self.xy[x]])
        return A


    def add_to_tree(self, x, prevx):
        self.S[x] = True
        self.prev[x] = prevx

        better_slack_idx = self.label_x[x] + self.label_y - self.weights[x] < self.slack
        self.slack[better_slack_idx] = self.label_x[x] + self.label_y[better_slack_idx] - self.weights[x, better_slack_idx]
        self.slackyx[better_slack_idx] = x

    def update_labels(self):
        delta = self.slack[np.logical_not(self.T)].min()
        self.label_x[self.S] -= delta
        self.label_y[self.T] += delta
        self.slack[np.logical_not(self.T)] -= delta

