from sklearn.svm import SVC
from sklearn.linear_model import SGDOneClassSVM
import numpy as np
from matrix import Matrix
from enumerations import Scale
import pandas as pd


class Darwin:
    fitness_functions = []

    def __init__(self):
        self.separator = SVC()
        self.fitness = None
        self.population = None
        self.target = None
        self.limit = 1000
        self.fitted = False
        self.target_mapping = lambda x: -1 if 'heal' in x else 1

    @staticmethod
    def margin_accuracy(margins):
        return np.round(sum(1 for i in margins if i > 0) / len(margins), 4)

    def get_margins(self, X, y, need_fit=False):
        if not self.fitted or need_fit:
            self.separator.fit(X, y)
            self.fitted = True
        return self.separator.decision_function(X) * y

    def spectra_margins(self, X, y, need_fit=False):
        return self.get_margins(X, y, need_fit)

    def download_population(self, path):
        population = pd.read_csv(path)
        y = population.pop(population.columns[0])
        y = y.apply(self.target_mapping)
        self.population = population
        self.target = y

    # @staticmethod
    # def plot_margins(X, y, margins, path='', cm=None):
    #     if not cm:
    #         cm = plt.cm.get_cmap('tab20')
    #     plt.figure(figsize=(20, 16))
    #     plt.axhline(0)
    #     lab = ''
    #     counter = 0
    #     for i, label in enumerate(y):
    #         if lab != label:
    #             lab = label
    #             counter += 1
    #             plt.plot(X[i, 0], margins[i], 'o', label=lab, color=cm.colors[counter])
    #         else:
    #             plt.plot(X[i, 0], margins[i], 'o', color=cm.colors[counter])
    #     if path:
    #         plt.savefig(path)
    #     else:
    #         plt.show()

if __name__ == '__main__':
    print('SYNTHESIS')
    d = Darwin()
    d.download_population('../tmp/EH_preproc.csv')
    
    from output import plot_margins
    plot_margins(d.population, d.target, d.get_margins(d.population, d.target))
    plot_margins(d.population, d.target, d.get_margins(d.population, d.target))








