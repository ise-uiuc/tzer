import matplotlib
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import pandas
import os
import re

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class Ploter:
    def __init__(self, cov_lim = None) -> None:
        self.fig, self.ax = plt.subplots(figsize=(9,6))
        # cov / time, cov / iteration, iteration / time
        self.cov_lim = cov_lim
        self.data = []

    def add(self, folder, nmax, name=None):
        path = os.path.join(folder, 'cov_by_time.txt')
        df = pandas.read_csv(path, usecols=[0, 1], header=None).to_numpy()
        self.data.append(df[-1,1])
        assert len(self.data) == nmax

    def plot(self, save='cov'):

        my_range = range(1, 11)
        plt.hlines(y=my_range, xmin=0, xmax=self.data, color='#007ACC', alpha=0.2, linewidth=6)

        plt.plot(self.data, my_range, "o", markersize=6, color='#007ACC', alpha=0.6)

        self.ax.set_xlabel('Edge Coverage in 4 hour', fontsize=15, fontweight='bold')
        self.ax.set_ylabel('$N_{max}$', fontsize=15, fontweight='bold')
        self.ax.set_xlim(left=self.cov_lim, right=30200)

        # set axis
        self.ax.tick_params(axis='both', which='major', labelsize=12)
        plt.yticks(my_range, my_range)

        # # add an horizonal label for the y axis 
        # self.fig.text(-0.23, 0.96, 'Transaction Type', fontsize=15, fontweight='black', color = '#333F4B')

        # change the style of the axis spines
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_bounds((1, len(my_range)))
        self.ax.spines['left'].set_position(('outward', 8))
        self.ax.spines['bottom'].set_position(('outward', 5))

        plt.tight_layout()
        plt.savefig(save + '.pdf')
        plt.savefig(save + '.png')

if '__main__' == __name__:
    ploter = Ploter(cov_lim = 25000)
    
    to_plot = []
    for p in os.listdir('.'):
        if os.path.isdir(p):
            ts = re.findall('rq3_3-tolerance-(\d+)_1-shrink-rand_gen', p)
            if ts:
                to_plot.append((p, int(ts[0])))

    for f, nmax in sorted(to_plot, key=lambda x: x[1]):
        # if nmax < 4:
        #     continue
        ploter.add(f, nmax, name=f'$N_{{max}}$={nmax}')
    ploter.plot('nmax_peak')
