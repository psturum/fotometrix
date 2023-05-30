import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

my_dict = {"Soja": 3.7, "Sødmælk": 3.4, "Mandel": 1.1, "Mini": 3.5}

df1 = pd.DataFrame(columns=['Mælk', 'Absorbance 1', 'Absorbance 2', 'dilution_grade', 'Gennemsnittet', 'mg protein', '% protein', 'Forventet', '%afvigelse'])


milk = widgets.Dropdown(
    options=['Soja', 'Sødmælk', 'Mandel', 'Mini'],
    value='Soja',
    disabled=False,
)
milk_label = widgets.Label(value='Mælk:')

abs_1 = widgets.BoundedFloatText(
    value=0.0,
    min=0,
    max=10.0,
    step=0.1,
)
abs_1_label = widgets.Label(value='Absorbans (1):')

abs_2 = widgets.BoundedFloatText(
    value=0.0,
    min=0,
    max=10.0,
    step=0.1,
)
abs_2_label = widgets.Label(value='Absorbans (2):')

dilution_grade = widgets.BoundedFloatText(
    value=0.0,
    min=0,
    max=10.0,
    step=0.1,
)
dilution_grade_label = widgets.Label(value='Dilution grade:')

button = widgets.Button(description='Tilføj data')
button1 = widgets.Button(description='Færdig')

# Define a grid layout
out = widgets.Output()
grid = widgets.GridspecLayout(8, 8)

# Add labels and inputs to the grid
grid[1, 0] = milk_label
grid[1, 1] = milk
grid[2, 0] = dilution_grade_label
grid[2, 1] = dilution_grade
grid[3, 0] = abs_1_label
grid[3, 1] = abs_1
grid[4, 0] = abs_2_label
grid[4, 1] = abs_2
grid[6, 0] = button
grid[6, 1] = button1

# Load the data from the csv file
data = pd.read_csv('../data/global.csv', usecols=['Koncentration','Gennemsnit'])

# Perform linear regression on Concentration and Avg_abs
y = data['Koncentration'].astype(float).values
x = data['Gennemsnit'].values
model = LinearRegression(fit_intercept=False).fit(x.reshape(-1, 1), y.reshape(-1, 1))

def add_row_to_df1(button):
    global df1
    row = {'Mælk': milk.value, 'Absorbance 1': abs_1.value, 'Absorbance 2': abs_2.value, 'dilution_grade': dilution_grade.value}
    df1 = pd.concat([df1, pd.DataFrame([row])], ignore_index=True)

    # Calculate the predicted mg protein using Gennemsnittet
    df1['Gennemsnittet'] = np.divide(df1['Absorbance 1'] + df1['Absorbance 2'], 2, out=np.zeros_like(df1['Absorbance 1']), where=df1['Absorbance 1'] + df1['Absorbance 2'] != 0)
    df1['mg protein'] = model.predict(df1[['Gennemsnittet']].values)

    # Calculate % protein and Forventet
    df1['% protein'] = df1['dilution_grade'] * df1['mg protein']
    df1['Forventet'] = df1['Mælk'].map(my_dict)

    # Calculate %afvigelse
    df1['%afvigelse'] = 100 - df1['Forventet'] / df1['% protein'] * 100

    with out:
        clear_output(wait=True)
        display(df1)

def upload_to_csv(button):
    df1.to_csv("../data/session2.csv",index=False)
    with out:
        clear_output()


# Register the button click event
button.on_click(add_row_to_df1)
button1.on_click(upload_to_csv)

def displaya():
    # Display the grid
    display(grid)
    display(out)