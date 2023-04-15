import matplotlib.pyplot as plt
import seaborn as sns

styles = plt.style.available
print(styles)

def plot_hist(X,xbins):
    for fig_style in styles:
        plt.style.use(fig_style)
        plt.hist(X,xbins)
        plt.xlabel('Delay period')
        plt.ylabel('# counts')
        plt.title('For all sessions')
        plt.xlim(min(1.1*X),max(1.05*X))
        plt.ylim(min(1.1*X),max(1.05*X))
        plt.show()