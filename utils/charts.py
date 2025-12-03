import matplotlib.pyplot as plt
import seaborn as sns

def corr_heatmap(corr):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    return fig
