# This source code is licensed under the license found in the
# LICENSE file in the {root}/privgem/tabular/ppgm directory of this source tree.
#
# This code has been modified from the version at
# https://github.com/BorealisAI/private-data-generation/tree/master/models/Private_PGM
# Modifications copyright (C) 2021-present, Kasra Hosseini.
#
# Copy of the header from https://github.com/BorealisAI/private-data-generation/blob/master/models/Private_PGM
#
# This source code is licensed under the license found in the
# LICENSE file in the {root}/models/Private_PGM/ directory of this source tree.
#
# This code has been modified from the original version at
# https://github.com/ryan112358/private-pgm/blob/master/examples/adult_example.py
# Modifications copyright (C) 2019-present, Royal Bank of Canada.

# Implementation of the Private-PGM generative model to generate private synthetic data
import numpy as np
from scipy import optimize, sparse
from .rdp_accountant import compute_rdp, get_privacy_spent
from .mbi import Dataset, FactoredInference, Domain

class PPGM:
    def __init__(self, target_variable, target_epsilon, target_delta):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.target_variable = target_variable
        self.model = None
        self.config = {}
    
    def create_config(self, df, exclude_cols=[]):
        for col in df.columns:
            if col in exclude_cols:
                continue
            col_count = len(df[col].unique())
            self.config[col] = col_count

    @staticmethod
    def moments_calibration(round1, round2, eps, delta):

        orders = range(2, 4096)

        def obj(sigma):
            rdp1 = compute_rdp(1.0, sigma / round1, 1, orders)
            rdp2 = compute_rdp(1.0, sigma / round2, 1, orders)
            rdp = rdp1 + rdp2
            privacy = get_privacy_spent(orders, rdp, target_delta=delta)
            return privacy[0] - eps + 1e-8

        low = 1.0
        high = 1.0
        while obj(low) < 0:
            low /= 2.0
        while obj(high) > 0:
            high *= 2.0
        sigma = optimize.bisect(obj, low, high)
        assert obj(sigma) - 1e-8 <= 0, 'not differentially private'  # true eps <= requested eps
        return sigma

    def train(self, train_df, config=None, iters=10000, cliques=None, exclude_cols=[]):
        if config is None:
            self.create_config(train_df, exclude_cols=exclude_cols)
        else:
            self.config = config
        domain = Domain(self.config.keys(), 
                        self.config.values())
        data = Dataset(train_df, domain)
        total = data.df.shape[0]

        if self.target_delta > 0:
            sigma = self.moments_calibration(1.0, 1.0, self.target_epsilon, self.target_delta)
        else:
            sigma = 1.0 / len(data.domain) / 2.0

        weights = np.ones(len(data.domain))
        weights /= np.linalg.norm(weights)  # now has L2 norm = 1

        measurements = []
        for col, wgt in zip(data.domain, weights):
            x = data.project(col).datavector()
            I = sparse.eye(x.size)
            if self.target_delta > 0:
                y = wgt * x + sigma * np.random.randn(x.size)
                measurements.append((I, y / wgt, 1.0 / wgt, (col,)))
            else:
                y = x + np.random.laplace(loc=0, scale=sigma, size=x.size)
                measurements.append((I, y, sigma, (col,)))

        # spend half of privacy budget to measure 2 way marginals with the target variable

        if cliques is None:
            cliques = []
            for col in data.domain:
                if col != self.target_variable:
                    cliques.append((col, self.target_variable))

        weights = np.ones(len(cliques))
        weights /= np.linalg.norm(weights)  # now has L2 norm = 1

        if self.target_delta == 0:
            sigma = 1.0 / len(cliques) / 2.0

        for cl, wgt in zip(cliques, weights):
            x = data.project(cl).datavector()
            I = sparse.eye(x.size)
            if self.target_delta > 0:
                y = wgt * x + sigma * np.random.randn(x.size)
                measurements.append((I, y / wgt, 1.0 / wgt, cl))
            else:
                y = x + np.random.laplace(loc=0, scale=sigma, size=x.size)
                measurements.append((I, y, sigma, cl))


        engine = FactoredInference(domain, log=True, iters=iters)
        self.model = engine.estimate(measurements, total=total, engine='MD')

    def generate(self, num_rows=None, output_type="dataframe", columns=None):
        syn_df = self.model.synthetic_data(rows=num_rows).df

        if output_type in ["numpy", "np", "array"]:
            X_syn = syn_df.drop([self.target_variable], axis=1).values
            y_syn = syn_df[self.target_variable].values
            return np.concatenate([X_syn, np.expand_dims(y_syn, axis=1)], axis=1)
        else:
            return syn_df
