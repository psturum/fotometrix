import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("../data/session3.csv", delimiter=',', usecols=["Mælk", "protein", "Ostemasse"])

# Set the positions of the bars on the x-axis
x = range(len(df['Mælk']))

def plot_():
    
    # Plotting the grouped bar plot
    fig, ax = plt.subplots()

    bar_width = 0.25
    ax.bar(x, df['protein'], width=bar_width, label='% protein', color='blue')
    ax.set_xlabel('Mælk')
    ax.set_ylabel('% protein', color='blue')
    ax2 = ax.twinx()
    ax2.bar([i + bar_width for i in x], df['Ostemasse'], width=bar_width, label='Ostemasse', color='green')
    ax2.set_ylabel('% Ostemasse', color='green')

    ax.set_xticks([i + bar_width/2 for i in x])
    ax.set_xticklabels(df['Mælk'])

    plt.title('Grouped Bar Plot - % protein and Ostemasse')
    plt.tight_layout()
    plt.show()
