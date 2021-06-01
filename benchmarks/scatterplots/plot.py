
##
import pandas as pd
import numpy as np
from scipy import stats, integrate
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import seaborn as sns
sns.set(color_codes=True)

##
def plotscatterphist(x, n1, y, n2, name, fit = True):
     with sns.axes_style("white"):
          d = {n1:x, n2:y}
          df = pd.DataFrame(data=d)
          g = sns.JointGrid(x=n1, y=n2, data=df)
          g = g.plot_joint(plt.hist2d,  bins=[50, 50],  cmap = 'Greys', norm=mc.LogNorm())
          g = g.plot_marginals(sns.distplot, color="gray", kde = fit)
          plt.xlabel(name, fontsize = 8)
          plt.colorbar()
          g.savefig(name+".pdf")
          plt.clf()


if __name__ == "__main__":

    name = 'copula'
    X = np.load(name+'.npy')
    x0 = [e[0] for e in X]
    x1 = [e[1] for e in X]
    x2 = [e[2] for e in X]
    plotscatterphist(x0, "1 marginal", x1, "2 marginal", name+"12", fit = False)
    plotscatterphist(x1, "2 marginal", x2, "3 marginal", name+"13", fit = False)

    name = 'transform'
    X = np.load(name+'.npy')
    x0 = [e[0] for e in X]
    x1 = [e[1] for e in X]
    x2 = [e[2] for e in X]
    x3 = [e[3] for e in X]
    plotscatterphist(x0, "1 marginal", x1, "2 marginal", name+"12", fit = False)
    plotscatterphist(x1, "2 marginal", x2, "3 marginal", name+"13", fit = False)
    plotscatterphist(x1, "2 marginal", x3, "4 marginal", name+"23", fit = False)
    plotscatterphist(x2, "3 marginal", x3, "4 marginal", name+"34", fit = False)
