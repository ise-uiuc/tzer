import matplotlib
import matplotlib.pyplot as plt
import pandas
import os
import time

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class Ploter:
    def __init__(self, cov_lim = None) -> None:
        self.legends = [] # type: ignore
        # cov / time, cov / iteration, iteration / time
        self.cov_lim = cov_lim

    def add(self, folder, name=None):
        path = os.path.join(folder, 'cov_by_time.txt')
        df = pandas.read_csv(path, usecols=[0, 1], header=None).to_numpy()
        
        plt.plot(df[:,0] / 3600, df[:,1], 'g', alpha=0.6, linewidth=6) # cov / time

        # if name:
        #     self.legends.append(name)
        # else:
        #     assert not self.legends

    def plot(self, save='cov'):
        # plt.legend(self.legends, prop={'weight':'bold'})
        plt.grid()
        
        if self.cov_lim is not None:
            plt.ylim(bottom=self.cov_lim)

        plt.xlabel('Time (Hour)',  fontweight='bold')
        plt.ylabel(ylabel='Edge Coverage',  fontweight='bold')
        
        # formatter = matplotlib.ticker.FuncFormatter(lambda ms, x: time.strftime('%M:%S', time.gmtime(ms // 3600)))
        # plt.gca().xaxis.set_major_formatter(formatter)
        # plt.title('Coverage $\\bf{Time}$ Efficiency')

        plt.xticks([0,4,8,12,16,20,24])
        plt.xlim(left=-0.3, right=24.3)
        plt.tight_layout()
        plt.savefig(save + '.pdf')
        plt.savefig(save + '.png')

if '__main__' == __name__:
    plt.figure(figsize=(15, 4))
    ploter = Ploter(cov_lim = 22000)
    
    for f in ['24-hr-run']:
        ploter.add(f, f)
    ploter.plot('24h')
