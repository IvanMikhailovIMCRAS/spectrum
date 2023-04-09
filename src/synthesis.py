from sklearn.svm import SVC
from sklearn.linear_model import SGDOneClassSVM
import numpy as np
from matrix import Matrix
from enumerations import Scale
import pandas as pd
from scipy.stats import expon, mode


class Darwin:
    fitness_functions = []
    epsilon = 0.01
    expon_scale = 2
    additional_transform = lambda x: x # lambda x: np.exp(-x) if x > 0 else -x**2

    def __init__(self):
        self.separator = SVC()
        self.fitness = None
        self.current_population = None
        self.current_target = None
        self.offspring = None
        self.limit = 1000
        self.fitted = False
        self.scale = None
        self.veclen = 0
        self.target_mapping = lambda x: -1 if 'heal' in x else 1
        self.proba_distr = None

    @staticmethod
    def margin_accuracy(margins):
        return np.round(sum(1 for i in margins if i > 0) / len(margins), 4)

    def get_margins(self, need_fit=False):
        X, y = self.current_population, self.current_target
        if not self.fitted or need_fit:
            self.separator.fit(X, y)
            self.fitted = True
        return self.separator.decision_function(X) * y

    def download_population(self, path):
        population = pd.read_csv(path)
        y = population.pop(population.columns[0])
        y = y.apply(self.target_mapping)
        self.current_population = population
        self.current_target = y
        self.scale = population.columns.astype(float)
        self.veclen = len(self.scale)

    @classmethod
    def breed(cls, f_body, s_body, f_label, s_label):

        choke_points = f_body - s_body
        indices = [i for i in range(len(choke_points)) if abs(choke_points[i]) < cls.epsilon]
        times = min(len(indices), np.ceil(expon.rvs(loc=1, scale=cls.expon_scale)))

        for _ in range(times):
            ind = np.random.randint(0, len(indices))
            choke = indices[ind]
            f_body = np.concatenate(f_body[:choke], s_body[choke:])
            s_body = np.concatenate(s_body[:choke], f_body[choke:])
            del indices[ind]
        if s_label == f_label:
            return s_body, s_label, f_body, f_label
        else:
            return s_body, None, f_body, None

    def __select_to_breed(self, distr_generator, max_ind=-1, **kwargs):
        if max_ind == -1:
            max_ind = len(self.current_population) - 1
        while True:
            f, s = np.round(distr_generator(size=2, **kwargs))
            if f <= max_ind and s <= max_ind:
                break
        return f, s

    def __sort_by_fitness(self):
        self.current_population['y'] = self.current_target
        self.current_population.sort_

    def form_generation(self, gen_size):
        ng = []
        new_labels = []






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
    # d = Darwin()
    # d.download_population('../tmp/EH_preproc.csv')
    # print(d.scale)
    values = []
    for _ in range(1000):
        values.append(expon.rvs(loc=1, scale=6))
    import matplotlib.pyplot as plt
    plt.hist(values, bins=100)
    plt.show()










