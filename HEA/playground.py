import pandas as pd
import numpy  as np

from bayes_opt.bayesian_optimization import BayesianOptimization

bo = BayesianOptimization(f=None, pbounds={})

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from oracle import Oracle


oracle = Oracle()


space = DesignSpace().parse([{'name' : 'Co', 'type' : 'num', 'lb' : 0.1, 'ub' : 0.3},
                             {'name' : 'Fe', 'type' : 'num', 'lb' : 0.1, 'ub' : 0.3},
                             {'name' : 'Mn', 'type' : 'num', 'lb' : 0.1, 'ub' : 0.3},
                             {'name' : 'V', 'type' : 'num', 'lb' : 0.1, 'ub' : 0.3}])


opt   = HEBO(space)


for i in range(100):
    rec = opt.suggest(n_suggestions = 1)
    opt.observe(rec, oracle(rec))
    print('After %d iterations, best obj is %.2f' % (i, opt.y.min()))