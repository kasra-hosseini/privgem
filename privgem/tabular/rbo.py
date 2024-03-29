# Python implementation of Rank-Biased Overlap (RBO) 
# 
# 
# * SOURCE: https://github.com/alan-turing-institute/QUIPP-pipeline
# which is licensed under MIT License
# 
# ---
# 
# * CREDIT: https://github.com/changyaochen/rbo
# 
# Copied from: 
# https://github.com/changyaochen/rbo
# which is licensed under MIT License
# 
# Copy of the license on https://github.com/changyaochen/rbo:
# 
# MIT License
# 
# Copyright (c) 2019 Changyao Chen
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

import pulp
pulp.LpSolverDefault.msg = 1
from pulp import LpMaximize, LpProblem, LpStatus, lpSum, LpVariable
from pulp import value, PULP_CBC_CMD

class RankingSimilarity(object):
    """
    This class will include some similarity measures between two different
    ranked lists.
    """

    def __init__(self, S, T, verbose=False):
        """
        Initialize the object with the required lists.
        Examples of lists:
        S = ['a', 'b', 'c', 'd', 'e']
        T = ['b', 'a', 1, 'd']

        Both lists relfect the ranking of the items of interest, for example,
        list S tells us that item 'a' is ranked first, 'b' is ranked second, etc.

        Args:
            S, T (list or numpy array): lists with alphanumeric elements. They
                could be of different lengths. Both of the them should be
                ranked, i.e., each element's position reflects its respective
                ranking in the list. Also we will require that there is no
                duplicate element in each list.
        """

        assert(type(S) in [list, np.ndarray])
        assert(type(T) in [list, np.ndarray])

        assert(len(S) == len(set(S)))
        assert(len(T) == len(set(T)))

        self.S, self.T = S, T
        self.N_S, self.N_T = len(S), len(T)
        self.verbose = verbose
        self.p = 0.5  # just a place holder

    def _bound_range(self, value):
        """Bounds the value to [0.0, 1.0].
        """

        try:
            assert(0 <= value <= 1 or np.isclose(1, value))
            return value

        except AssertionError:
            print('Value out of [0, 1] bound, will bound it.')
            larger_than_zero = max(0.0, value)
            less_than_one = min(1.0, larger_than_zero)
            return less_than_one

    def rbo(self, k=None, p=1.0, ext=False):
        """
        This the weighted non-conjoint measures, namely, rank-biased overlap.
        Unlike Kendall tau which is correlation based, this is intersection
        based.
        The implementation if from Eq. (4) or Eq. (7) (for p != 1) from the RBO
        paper: http://www.williamwebber.com/research/papers/wmz10_tois.pdf

        If p=1, it returns to the un-bounded set-intersection overlap, according
        to Fagin et al.
        https://researcher.watson.ibm.com/researcher/files/us-fagin/topk.pdf

        The fig. 5 in that RBO paper can be used as test case.
        Note there the choice of p is of great importance, since it essentically
        control the 'top-weightness'. Simply put, to an extreme, a small p value
        will only consider first few items, whereas a larger p value will
        consider more itmes. See Eq. (21) for quantitative measure.

        Args:
            k (int), default None: The depth of evaluation.
            p (float), default 1.0: Weight of each agreement at depth d:
                p**(d-1). When set to 1.0, there is no weight, the rbo returns
                to average overlap.
            ext (Boolean) default False: If True, we will extropulate the rbo,
                as in Eq. (23)

        Returns:
            The rbo at depth k (or extrapolated beyond)
        """

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        if k is None:
            k = float('inf')
        k = min(self.N_S, self.N_T, k)

        # initilize the agreement and average overlap arrays
        A, AO = [0] * k, [0] * k
        if p == 1.0:
            weights = [1.0 for _ in range(k)]
        else:
            assert(0.0 < p < 1.0)
            weights = [1.0 * (1 - p) * p**d for d in range(k)]

        self.p = p

        # using dict for O(1) look up
        S_running, T_running = {self.S[0]: True}, {self.T[0]: True}
        A[0] = 1 if self.S[0] == self.T[0] else 0
        AO[0] = weights[0] if self.S[0] == self.T[0] else 0

        PP = ProgressPrintOut(k) if self.verbose else NoPrintOut()
        for d in range(1, k):

            PP.printout(d, delta=1)
            tmp = 0
            # if the new item from S is in T already
            if self.S[d] in T_running:
                tmp += 1
            # if the new item from T is in S already
            if self.T[d] in S_running:
                tmp += 1
            # if the new items are the same, which also means the previous
            # two cases didn't happen
            if self.S[d] == self.T[d]:
                tmp += 1

            # update the agreement array
            A[d] = 1.0 * ((A[d - 1] * d) + tmp) / (d + 1)

            # update the average overlap array
            if p == 1.0:
                AO[d] = ((AO[d - 1] * d) + A[d]) / (d + 1)
            else:  # weighted average
                AO[d] = AO[d - 1] + weights[d] * A[d]

            # add the new item to the running set (dict)
            S_running[self.S[d]] = True
            T_running[self.T[d]] = True

        if ext and p < 1:
            return self._bound_range(AO[-1] + A[-1] * p**k)
        else:
            return self._bound_range(AO[-1])

    def rbo_ext(self, p=0.98):
        """
        This is the ultimate implementation of the rbo, namely, the extrapolated
        version. The corresponding formula is Eq. (32) in the rbo paper
        """

        assert(0.0 < p < 1.0)
        self.p = p

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        # since we are dealing with un-even lists, we need to figure out the
        # long (L) and short (S) list first. The name S might be confusing
        # but in this function, S refers to short list, L refers to long list
        if len(self.S) > len(self.T):
            L, S = self.S, self.T
        else:
            S, L = self.S, self.T

        s, l = len(S), len(L)

        # initilize the overlap and rbo arrays
        # the agreement can be simply calculated from the overlap
        X, A, rbo = [0] * l, [0] * l, [0] * l

        # first item
        S_running, L_running = {S[0]}, {L[0]}  # for O(1) look up
        X[0] = 1 if S[0] == L[0] else 0
        A[0] = X[0]
        rbo[0] = 1.0 * (1 - p) * A[0]

        # start the calculation
        PP = ProgressPrintOut(l) if self.verbose else NoPrintOut()
        disjoint = 0
        ext_term = A[0] * p

        for d in range(1, l):
            PP.printout(d, delta=1)

            if d < s:  # still overlapping in length

                S_running.add(S[d])
                L_running.add(L[d])

                # again I will revoke the DP-like step
                overlap_incr = 0  # overlap increament at step d

                # if the new itmes are the same
                if S[d] == L[d]:
                    overlap_incr += 1
                else:
                    # if the new item from S is in L already
                    if S[d] in L_running:
                        overlap_incr += 1
                    # if the new item from L is in S already
                    if L[d] in S_running:
                        overlap_incr += 1

                X[d] = X[d - 1] + overlap_incr
                # Eq. (28) that handles the tie. len() is O(1)
                A[d] = 2.0 * X[d] / (len(S_running) + len(L_running))
                rbo[d] = rbo[d - 1] + 1.0 * (1 - p) * (p**d) * A[d]

                ext_term = 1.0 * A[d] * p**(d + 1)  # the extropulate term

            else:  # the short list has fallen off the cliff
                L_running.add(L[d])  # we still have the long list

                # now there is one case
                overlap_incr = 1 if L[d] in S_running else 0

                X[d] = X[d - 1] + overlap_incr
                A[d] = 1.0 * X[d] / (d + 1)
                rbo[d] = rbo[d - 1] + 1.0 * (1 - p) * (p**d) * A[d]

                X_s = X[s - 1]  # this the last common overlap
                # second term in first parenthesis of Eq. (32)
                disjoint += 1.0 * (1 - p) * (p**d) * \
                    (X_s * (d + 1 - s) / (d + 1) / s)
                ext_term = 1.0 * ((X[d] - X_s) / (d + 1) + X[s - 1] / s) * \
                    p**(d + 1)  # last term in Eq. (32)

        return self._bound_range(rbo[-1] + disjoint + ext_term)

    def top_weightness(self, p=None, d=None):
        """
        This function will evaluate the degree of the top-weightness of the rbo.
        It is the implementation of Eq. (21) of the rbo paper.

        As a sanity check (per the rbo paper),
        top_weightness(p=0.9, d=10) should be 86%
        top_weightness(p=0.98, d=50) should be 86% too

        Args:
            p (float), defalut None: A value between zero and one.
            d (int), default None: Evaluation depth of the list.

        Returns:
            A float between [0, 1], that indicates the top-weightness.
        """

        # sanity check
        if p is None:
            p = self.p
        assert (0. < p < 1.0)

        if d is None:
            d = min(self.N_S, self.N_T)
        else:
            d = min(self.N_S, self.N_T, int(d))

        if d == 0:
            top_w = 1
        elif d == 1:
            top_w = 1 - 1 + 1.0 * (1 - p) / p * (np.log(1.0 / (1 - p)))
        else:
            sum_1 = 0
            for i in range(1, d):
                sum_1 += 1.0 * p**(i) / i
            top_w = 1 - p**(i) + 1.0 * (1 - p) / p * (i + 1) * \
                (np.log(1.0 / (1 - p)) - sum_1)  # here i == d-1

        if self.verbose:
            print('The first {} ranks have {:6.3%} of the weight of '
                  'the evaluation.'.format(d, top_w))

        return self._bound_range(top_w)

    def correlated_rank_similarity(self, correlation_matrix, k=None, p=1.0, ext=False):
        """
        RBO adapted to use doubly-stochastic correlated rank similarity
        instead of the overlap used in the RBO paper.

        Args:
            correlation_matrix (pandas.DataFrame):
                The correlation or strength of association between all features
            k (int), default None:
                The depth of evaluation.
            p (float), default 1.0: Weight of each agreement at depth d:
                p**(d-1). When set to 1.0, there is no weight, returns
                to average "correlated overlap".
            ext (Boolean) default False: If True, we will extrapolate the correlated rank,
                as in Eq. (23) of the original RBO paper.

        Returns:
            The correlated rank similarity at depth k (or extrapolated beyond)
        """

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        self.p = p

        if k is None:
            k = float('inf')
        k = min(self.N_S, self.N_T, k)

        X, A, AO = np.zeros(k), np.zeros(k), np.zeros(k)

        if p == 1.0:
            weights = np.ones(k)
        else:
            assert (0.0 < p < 1.0), "p not >0.0 and <1.0"
            weights = np.array([1.0 * (1 - p) * p ** d for d in range(k)])

        for d in range(0, k):
            X[d], _ = self.lp_optimiser(d, correlation_matrix)

            A[d] = X[d] / (d + 1)

            if p == 1.0:  # equivalent to average overlap in equation 4 of the RBO paper
                AO[d] = A[:(d + 1)].sum() / (d + 1)
            else:  # weighted average - i.e. equivalent to RBO in equation 7 of the RBO paper
                AO[d] = (A[:(d + 1)] * weights[:(d + 1)]).sum()

        if ext and p < 1:
            return self._bound_range(AO[-1] + A[-1] * p ** k)
        else:
            return self._bound_range(AO[-1])

    def correlated_rank_similarity_ext(self, correlation_matrix, p=0.98):
        """
        Extrapolated RBO adapted to use doubly-stochastic correlated rank similarity
        instead of the overlap used in the RBO paper. This version implements an equivalent
        metric to equation 32 in the RBO paper, where uneven lists and ties are taken into
        account.

        Args:
            correlation_matrix (pandas.DataFrame):
                The correlation or strength of association between all features
            p (float), default 0.98: Weight of each agreement at depth d:
                p**(d-1). When set to 1.0, there is no weight, returns
                to average "correlated overlap".

        Returns:
            The extrapolated correlated rank similarity
        """

        assert (0.0 < p < 1.0), "p is not >0.0 and <1.0"
        self.p = p

        if not self.N_S and not self.N_T:
            return 1  # both lists are empty

        if not self.N_S or not self.N_T:
            return 0  # one list empty, one non-empty

        # since we are dealing with un-even lists, we need to figure out the
        # long (L) and short (S) list first. The name S might be confusing
        # but in this function, S refers to short list, L refers to long list
        if len(self.S) > len(self.T):
            L, S = self.S, self.T
        else:
            S, L = self.S, self.T

        s, l = len(S), len(L)

        # initilize the overlap and rbo arrays
        # the agreement can be simply calculated from the overlap
        X, A, rbo = np.zeros(l), np.zeros(l), np.zeros(l)
        weights = np.array([1.0 * (1 - p) * p ** d for d in range(l)])
        disjoint = 0

        # start the calculation
        for d in range(1, l):

            if d < s:  # still overlapping in length

                X[d], _ = self.lp_optimiser(d, correlation_matrix)

                # Eq. (28) that handles the tie. len() is O(1)
                A[d] = 2.0 * X[d] / (len(S[:(d + 1)]) + len(L[:(d + 1)]))

                rbo[d] = (weights[:(d + 1)] * A[:(d + 1)]).sum()

                ext_term = 1.0 * A[d] * p ** (d + 1)  # the extrapolate term

            else:  # the short list has fallen off the cliff

                X[d], _ = self.lp_optimiser(d, correlation_matrix)

                A[d] = 1.0 * X[d] / (d + 1)

                rbo[d] = (weights[:(d + 1)] * A[:(d + 1)]).sum()

                X_s = X[s - 1]  # this the last common overlap

                # second term in first parenthesis of Eq. (32)
                disjoint += 1.0 * (1 - p) * (p ** d) * (X_s * (d + 1 - s) / (d + 1) / s)

                ext_term = 1.0 * ((X[d] - X_s) / (d + 1) + X[s - 1] / s) * \
                           p ** (d + 1)  # last term in Eq. (32)

        return self._bound_range(rbo[-1] + disjoint + ext_term)

    def lp_optimiser(self, d, correlation_matrix):
        """Solves the doubly stochastic LP problem given used in the
        correlated rank metric.

        Args:
            d (int):
                The depth of evaluation.
            correlation_matrix (pandas.DataFrame):
                The correlation or strength of association between all features

        Returns:
            score (float):
                The optimal correlated rank score at depth d
            opt_W (np.array):
                The optimal doubly stochastic matrix W

        """

        # if d is larger than the length of the smaller list
        # this happens when working with uneven lists and want
        # to explore them fully
        if self.N_S != self.N_T and d >= min(self.N_S, self.N_T):
            if self.N_S > self.N_T:
                d1 = d
                d2 = small_len = self.N_T - 1
            else:
                d1 = small_len = self.N_S - 1
                d2 = d
        else:
            d1 = d2 = small_len = d

        # Define the LP problem
        prob = LpProblem("Doubly-stochastic", LpMaximize)
        variable_names = [self.S[i] + "_" + self.T[j]
                          for i in range(d1 + 1)
                          for j in range(d2 + 1)]
        W = LpVariable.matrix("X", variable_names, cat='Continuous', lowBound=0, upBound=1)
        W_np = np.array(W).reshape(d1 + 1, d2 + 1)

        # Reorder the correlations
        corr_subset = np.eye(small_len + 1)
        # if uneven lists and we are beyond the maximum index of the small one,
        # then add rows or columns to the corr_subset matrix
        if self.N_S != self.N_T and d >= min(self.N_S, self.N_T):
            if self.N_S > self.N_T:
                corr_subset = np.vstack((corr_subset, np.zeros((d1 - d2, small_len + 1))))
            else:
                corr_subset = np.hstack((corr_subset, np.zeros((small_len + 1, d2 - d1))))

        for i, var1 in enumerate(self.S[:(d1 + 1)]):
            for j, var2 in enumerate(self.T[:(d2 + 1)]):
                corr_subset[i][j] = correlation_matrix[var1][var2]

        # Objective function
        obj_func = lpSum(W_np * corr_subset ** 0.5), "main objective"
        prob += obj_func

        # Constraints
        for i in range(d1 + 1):
            prob += lpSum(W_np[i][j] for j in range(d2 + 1)) == 1, "Double stochastic row" + str(i)
        for i in range(d2 + 1):
            prob += lpSum(W_np[j][i] for j in range(d1 + 1)) == 1, "Double stochastic col" + str(i)

        # Solve and print result
        prob.solve(PULP_CBC_CMD(msg=False))
        #print(LpStatus[prob.status])

        # Get W and score
        opt_W = np.array([v.varValue for v in W]).reshape(d1 + 1, d2 + 1)
        opt_score = value(prob.objective)

        return opt_score, opt_W


class ProgressPrintOut(object):
    """Quick status print out.
    """

    def __init__(self, N):
        self._old = 0
        self._total = N

    def printout(self, i, delta=10):
        # print out progess every delta %
        cur = 100 * i // self._total
        if cur >= self._old + delta:
            print('\r', 'Current progress: {} %...'.format(cur), end='')
            self._old = cur
        if i == self._total - 1:
            print('\r', 'Current progress: 100 %...', end='')
            print('\nFinished!')

        return


class NoPrintOut(object):
    def printout(self, i, delta=10):
        pass

