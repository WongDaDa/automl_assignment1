import numpy as np
import pandas as pd
import unittest

from solution import AverageRank, GreedyDefaults


class TestMetaModels(unittest.TestCase):

    def test_on_toy_data(self):
        with open('data_toy.csv', 'r') as fp:
            data = pd.read_csv(fp).set_index(['configuration'])

        avg_rank_model = AverageRank()
        avg_ranks = avg_rank_model.fit(data, None)

        avg_ranks_correct = ['theta0', 'theta1', 'theta2', 'theta3']
        self.assertEqual(avg_ranks_correct, avg_ranks)

        greedy_defaults_model = GreedyDefaults()
        greedy_defaults = greedy_defaults_model.fit(data, np.sum)
        greedy_defaults_correct = ['theta0', 'theta3', 'theta1']
        self.assertEqual(greedy_defaults_correct, greedy_defaults)

    def test_on_svm_data(self):
        with open('data_svm.csv', 'r') as fp:
            data = pd.read_csv(fp).set_index(['kernel_rbf', 'kernel_poly', 'kernel_linear', 'c', 'gamma', 'degree'])

        avg_rank_model = AverageRank()
        avg_ranks = avg_rank_model.fit(data, None)
        avg_best_config = (1.0, 0.0, 0.0, 0.8333333333333334, -0.3252574989159953, 0.0)
        avg_best_config_ttt = 0.989583
        self.assertEqual(avg_best_config, avg_ranks[0])
        self.assertEqual(avg_best_config_ttt, avg_rank_model.evaluate(avg_ranks[0:1], data['perf-on-tic-tac-toe']))

        greedy_defaults_model = GreedyDefaults()
        greedy_defaults = greedy_defaults_model.fit(data, np.sum)
        greedy_best_config = (1.0, 0.0, 0.0, 1.0, -0.25, 0.0)
        greedy_best_config_ttt = 1.0
        self.assertEqual(greedy_best_config, greedy_defaults[0])
        self.assertEqual(greedy_best_config_ttt,
                         greedy_defaults_model.evaluate(greedy_defaults[0:1], data['perf-on-tic-tac-toe']))

