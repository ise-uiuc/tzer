import matplotlib
import matplotlib.pyplot as plt
import pandas
import os

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
# plt.rc('figure', titlesize=16)  # fontsize of the figure title

import matplotlib.pyplot as plt
import pandas
import os

class Ploter:
    def __init__(self, cov_lim = None) -> None:
        self.legends = [] # type: ignore
        # cov / time, cov / iteration, iteration / time
        fig, axs = plt.subplots(1, 3, figsize=(16, 4))
        self.fig = fig 
        self.axs = axs
        self.cov_lim = cov_lim

    def add(self, folder, name=None):
        path = os.path.join(folder, 'cov_by_time.txt')
        df = pandas.read_csv(path, usecols=[0, 1], header=None).to_numpy()
        
        self.axs[0].plot(df[:,0], df[:,1], alpha=0.6, linewidth=4) # cov / time
        self.axs[1].plot(range(len(df[:,0])), df[:,1], alpha=0.6, linewidth=4) # cov / iteration
        self.axs[2].plot(df[:,0], range(len(df[:,1])), alpha=0.6, linewidth=4) # iter / time

        if name:
            self.legends.append(name)
        else:
            assert not self.legends

    def plot(self, save='cov'):
        for axs in self.axs:
            axs.legend(self.legends, prop={'weight':'bold'})
        # plt.legend(self.legends)
        
        if self.cov_lim is not None:
            self.axs[0].set_ylim(bottom=self.cov_lim)
            self.axs[1].set_ylim(bottom=self.cov_lim)

        self.axs[0].set(
            xlabel=r'$\bf{Time / Second}$',
            ylabel=r'$\bf{\# Coverage}$')
        self.axs[0].set_title(r'$\bf{Coverage\ Time\ Efficiency}$')

        self.axs[1].set(
            ylabel=r'$\bf{\# Coverage}$',
            xlabel=r'$\bf{\# Iteration}$')
        self.axs[1].set_title(r'$\bf{Coverage\ Iteration\ Efficiency}$')

        self.axs[2].set(
            xlabel=r'$\bf{Time / Second}$',
            ylabel=r'$\bf{\# Iteration}$')
        self.axs[2].set_title(r'$\bf{Iteration\ Speed}$')
        
        for axs in self.axs:
            axs.grid(True)

        plt.tight_layout()
        plt.savefig(save + '.pdf')
        plt.savefig(save + '.png')

if '__main__' == __name__:
    ploter = Ploter(cov_lim = 20000)
    
    for f in ['Tzer without seeds', 'Tzer with all seeds']:
        ploter.add(f, f)
    ploter.plot('seeds_complete')
