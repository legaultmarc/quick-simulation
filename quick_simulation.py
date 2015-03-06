#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
from collections import OrderedDict, namedtuple


# This can be used when there is an unsimulated variable to add to the model.
Variable = namedtuple("Variable", ["name", "x"])


class GeneticVariable(object):
    """Simulate a genetic variable.

    :param name: Name of the variable (e.g. SNP name).
    :param n: Number of samples to simulate.
    :param effect: The effect size (beta).
    :param maf: The minor allele frequency for the simulation. Genotypes will
                then be simulated from Hardy Weinberg frequencies.
    :param dosage_var: Variance or noise to add to the dosage simulated
                       values.
    :param call_rate: Simulate no-calls at the frequency `1-call_rate` using
                      `np.nan` values.

    """
    def __init__(self, name, n, effect, maf, dosage_var=None, call_rate=1):
        self.name = name
        self.n = n
        self.effect = effect
        self.maf = maf
        self.dosage_var = dosage_var
        self.call_rate = call_rate
        
        self.x = self._build_genotypes()
        
    def _build_genotypes(self):
        """Build a vector of genotypes or dosage."""
        x = np.zeros(self.n)
        
        # Frequencies derived from HWE.
        num_hetero = 2 * self.maf * (1 - self.maf) * self.n
        num_homo_minor = self.maf ** 2 * self.n
        
        x[:num_hetero] = 1
        x[num_hetero:num_hetero+num_homo_minor] = 2
        np.random.shuffle(x)
        
        # Add noise for dosage values if needed.
        if self.dosage_var:
            x[x == 0] += np.abs(
                np.random.normal(0, self.dosage_var, len(x[x == 0]))
            )
            x[x == 1] += np.random.normal(0, self.dosage_var, len(x[x == 1]))
            x[x == 2] -= np.abs(
                np.random.normal(0, self.dosage_var, len(x[x == 2]))
            )

        # Mask some values if the call rate is not 1.
        if self.call_rate < 1:
            missing_rate = 1 - self.call_rate
            missing_number = missing_rate * self.n
            missing_idx = np.arange(0, self.n)
            np.random.shuffle(missing_idx)
            missing_idx = missing_idx[:missing_number]
            x[missing_idx] = np.nan
        
        return x
    
    def __repr__(self):
        return self.name
        
        
class Simulation(object):
    """A Simulation object that computes the outcome from the underlying model.

    :param predictors: A list of predictor objects. These need to have a `name`
                       and an `x` attributes containing a name for the
                       parameter and a numpy vector of values, respectively.
    :param outcome_type: Either _discrete_ or _continuous_.
    :param interactions: A dict of pairs of variables to effect size. _e.g._
                         `{(snp1, snp2): 0.3}` where `snp1` and `snp2` could
                         be instances of :py:class:`GeneticVariable`.
    :param noise: Variance to add to the simulated samples.
    :param intercept: An intercept value for continuous traits.

    """
    def __init__(self, predictors, outcome_type="discrete",
                 interactions=None, noise=0, intercept=0):

        self.model = "y ~ {}".format(
            " + ".join([repr(e) for e in predictors]),
        )

        if interactions:
            self.model += " + "
            self.model += " + ".join(
                ["{} x {}".format(*k) for k in interactions.keys()]
            )

        self.predictors = predictors
        if outcome_type not in ("discrete", "continuous"):
            msg = "Invalid outcome_type '{}'. Authorized values are {}"
            msg = msg.format(outcome_type, ("discrete", "continuous"))
            raise Exception(msg)

        self.outcome_type = outcome_type
        self.intercept = intercept
        self.interactions = interactions
        self.noise = noise
        self.y = self._build_outcome()
    
    def _build_outcome(self):
        assert len(set([e.n for e in self.predictors])) == 1
        n = self.predictors[0].n
        y = np.zeros(n)
        for i in range(len(y)):
            for pred in self.predictors:
                y[i] += pred.effect * pred.x[i]
            if self.interactions:
                for preds, effect in self.interactions.items():
                    pred1, pred2 = preds
                    y[i] += effect * pred1.x[i] * pred2.x[i]

        if self.outcome_type == "discrete":
            # We discretize using the logistic function.
            y = np.round(1 / (1 + np.exp(-y)))
        
        # Special parameters for continuous traits.
        else:
            if self.noise > 0:
                y += np.random.normal(0, self.noise, n)
            
            y += self.intercept

        return y

    def plot_main_effects(self, figsize=(15, 5)):
        """Plot the main effects.

        Create a single-row plot representing the outcome as a function of
        all of the predictor variables.

        """

        import matplotlib.pyplot as plt
        import seaborn as sbn

        fig, axes = plt.subplots(1, len(self.predictors),
                                 figsize=figsize, sharey=True)
        for i, ax in enumerate(axes):
            if self.outcome_type == "discrete":
                n = len(self.predictors[i].x)
                ax.scatter(
                    self.predictors[i].x,
                    self.y + np.random.normal(0, 0.02, n)
                )
            else:
                ax.scatter(self.predictors[i].x, self.y)
            ax.set_xlabel(self.predictors[i])
            if i == 0:
                ax.set_ylabel("Outcome")
    
    def to_dataframe(self):
        """Dumps the simulated data to a pandas DataFrame.

        This is convenient for use with popular Python statistical packages
        or to simply write a CSV file containing the simulated data to disk.

        """
        import pandas as pd

        d = OrderedDict({"y": self.y})
        for pred in self.predictors:
            d[pred.name] = pred.x
            
        df = pd.DataFrame(d)
        return df
