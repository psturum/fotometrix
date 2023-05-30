import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

my_dict = {"Soja": 3.7, "Sødmælk": 3.4, "Mandel": 1.1, "Mini": 3.5, "Havre": 0.7, "Ris": 0.0}

milk = widgets.Dropdown(
    options=['Soja', 'Sødmælk', 'Mandel', 'Mini'],
    value='Soja',
    disabled=False,
)
milk_label = widgets.Label(value='Mælk:')

weight_1_label = widgets.Label(value='Vægt (før):')

weight_1 = widgets.BoundedIntText(
    value=0,
    min=0,
    max=2000,
    step=1,
)

weight_2_label = widgets.Label(value='Vægt (efter):')

weight_2 = widgets.BoundedIntText(
    value=0,
    min=0,
    max=2000,
    step=1,
)

button = widgets.Button(description='Tilføj data')
button1 = widgets.Button(description='Færdig')

# Load the data from the csv file
data = pd.read_csv('../data/global2.csv', usecols=['Mælk','protein'])
data = data.groupby('Mælk').mean().reset_index()
data['Ostemasse'] = [0,0,0,0]

def add_row_to_df1(button):
    global data

    if weight_1.value != 0 and weight_2.value != 0:
        data.loc[data['Mælk'] == milk.value, 'Ostemasse'] = weight_2.value / weight_1.value
    else:
        data.loc[data['Mælk'] == milk.value, 'Ostemasse'] = 0.0

    with out:
        clear_output(wait=True)
        display(data)

def upload_to_csv(button):
    data.to_csv("../data/session3.csv",index=False)
    with out:
        clear_output(wait=True)

button.on_click(add_row_to_df1)
button1.on_click(upload_to_csv)
# Define a grid layout
out = widgets.Output()
grid = widgets.GridspecLayout(8, 8)

grid[1, 0] = milk_label
grid[1, 1] = milk
grid[2, 0] = weight_1_label
grid[2, 1] = weight_1
grid[3, 0] = weight_2_label
grid[3, 1] = weight_2
grid[5, 0] = button
grid[5, 1] = button1

def display_():
    display(grid)
    display(out)