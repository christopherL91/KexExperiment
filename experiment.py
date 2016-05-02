#!/usr/bin/env python2

# The MIT License (MIT)
# Copyright (c) 2016 Samuel Philipson, Christopher Lillthors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function
import graphlab as gl
import os
from time import time
import csv
import datetime
import numpy as np

timestamp = str(datetime.datetime.now())
output_path = os.path.join('.', timestamp)

def f1_score(precision, recall):
    return 2 * ((precision * recall)/(precision + recall))

def evaluate():
    #   Swap these depending on dataset.
    ratings = gl.SFrame.read_csv(os.path.join('.', 'ratings.dat'),
        delimiter='::',
    	column_type_hints=[int,int,float,int])
    #ratings = gl.SFrame.read_csv(os.path.join('.', 'small-ratings.csv'),
    #        column_type_hints=[int,int,float,int])
    num_ratings = ratings.num_rows()
    print('There are {} number of ratings in the dataset'.format(num_ratings))

    k = 4
    folds = gl.cross_validation.KFold(ratings, k)

    #   Evaluation section
    #   -------------------------------------
    seed = 5L
    iterations = 50
    ranks = [4]
    #ranks = range(4, 84, 4) #   Change this
    solver = 'als'  # als or sgd
    verbose = False

    min_error = float('inf')
    best_rank = -1
    best_iteration = -1

    with open(output_path + '/' + solver + '-' + str(iterations) + '-' + 'out.csv', 'w') as f:
        fieldnames = ['rmse', 'time', 'rank', 'precision', 'recall', 'f1-score']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank in ranks:
            error = 0
            time_total = 0
	    precision = 0
            recall = 0
            for train, valid in folds:
                t0 = time()
                model = gl.recommender.factorization_recommender.create(train,
                                                             user_id='userId',
                                                             item_id='movieId',
                                                             target='rating',
                                                             solver=solver,
                                                             max_iterations=iterations,
                                                             random_seed=seed,
                                                             num_factors=rank,
                                                             verbose=verbose)
                #   Stop clock and get execution time.
                time_total += round(time() - t0, 3)
                error += model.evaluate_rmse(valid, target='rating')['rmse_overall']
                precision_recall = model.evaluate_precision_recall(valid)['precision_recall_overall'].to_numpy()
		print(precision_recall)
                precision += precision_recall[0,1]
                recall += precision_recall[0,2]
            error /= k
            time_total /= k
            precision /= k
            recall /= k
            #   Write to file
            writer.writerow({'rmse': error,
                             'time': time_total,
                             'rank': rank,
                             'precision': precision,
                             'recall': recall,
                             'f1-score': f1_score(precision, recall)})
            print('For rank {} the RMSE is {}'.format(rank, error))
            if error < min_error:
                min_error = error
                best_rank = rank
    print('The best model was trained with rank {}'.format(best_rank))
if __name__ == '__main__':
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    evaluate()
