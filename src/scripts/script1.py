import ipywidgets as widgets
import warnings
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, HTML, clear_output
from sklearn.linear_model import LinearRegression
import pandas as pd
import re

df1 = pd.read_csv('../data/session.csv')
df = pd.DataFrame(columns=['Koncentration','Gennemsnit', 'Gruppe'])

def blah():
    # Create widgets
    gruppe = widgets.Dropdown(
        options=['1', '2', '3', '4', '5'],
        value='1',
        disabled=False,
        layout=widgets.Layout(width='200px')
    )
    gruppe_label = widgets.Label(value='Gruppe:', layout=widgets.Layout(width='100px'))

    # Create widgets
    concentration = widgets.Dropdown(
        options=['0.0', '0.1', '0.2', '0.3', '0.4'],
        value='0.0',
        disabled=False,
        layout=widgets.Layout(width='200px')
    )
    concentration_label = widgets.Label(value='Protein indhold:', layout=widgets.Layout(width='100px'))

    abs1_label = widgets.Label(value='Absorbans data:', layout=widgets.Layout(width='200px'))
    abs_1 = widgets.Textarea(
        placeholder='Indsæt decimal tal separeret af kommaer',
        rows=5,
        layout=widgets.Layout(width='200px')
    )

    index_dropdown = widgets.Dropdown(
        options=[],
        description='Vælg række:',
        layout=widgets.Layout(width='80%')
    )

    button = widgets.Button(description="Tilføj Data")
    button1 = widgets.Button(description="Upload Data")
    button2 = widgets.Button(description="Fjern data")

    # Function to remove a row from the DataFrame based on the index
    def remove_row_by_index(button):
        index = index_dropdown.value
        try:
            df.drop(index=index, inplace=True)
            index_dropdown.options = df.index
            with out:
                clear_output(wait=True)
                display(df)
            print(f"Row at index {index} removed successfully.")
        except KeyError:
            print(f"Error: Index {index} does not exist in the DataFrame.")

    def add_row_to_df1(button):
        global df1
        global df

        row = {'Koncentration': concentration.value, 'Gennemsnit': calculate_average(abs_1.value), 'Gruppe': gruppe.value}
        df1 = pd.concat([df1, pd.DataFrame([row])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        index_dropdown.options = df.index
        index_dropdown.value = index_dropdown.options[0]

        with out:
            clear_output(wait=True)
            display(df)

    def upload_to_csv(button):
        df1.to_csv("../data/session.csv", index=False)
        with out:
            clear_output(wait=True)

    def calculate_average(numbers):
        numbers = re.sub(r'\s', '', numbers)
        number_list = numbers.split(',')
        number_list = [float(num) for num in number_list]
        
        try:
            average = np.mean(number_list)
        except ZeroDivisionError:
            return None  # Handle the division by zero case appropriately
            
        return average

    button.on_click(add_row_to_df1)
    button1.on_click(upload_to_csv)
    button2.on_click(remove_row_by_index)

    out = widgets.Output()
    grid = widgets.GridspecLayout(11, 6, width='500px')

    grid[1, 0] = gruppe_label
    grid[1, 1] = gruppe
    grid[2, 0] = concentration_label
    grid[2, 1] = concentration
    grid[3, 0] = abs1_label
    grid[3:5, 1] = abs_1
    grid[6, 0] = button
    grid[10, 0] = button1
    grid[8, 1] = button2
    grid[8, 0] = index_dropdown
    with out:
        display(df)
    display(grid)
    display(out)
    
    
    
def linear_regression_widget():
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    out = widgets.Output()

    y = df1['Koncentration'].astype(float).values
    x = df1['Gennemsnit'].values
    model = LinearRegression(fit_intercept=False).fit(x.reshape(-1, 1), y.reshape(-1, 1))

    x_range = np.linspace(0, x.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))
    plt.plot(x_range, y_range, color='r')
    plt.scatter(x, y)

    # Add labels and title to the plot
    plt.xlabel('Absorbans')
    plt.ylabel('Koncentration i mg')
    plt.title('Linear Regression Model')
    with out:
        out.clear_output(wait=True)
        plt.show()

    # create input widget
    inpt_label = widgets.Label(value='Absorbans(x): ')
    input_widget = widgets.BoundedFloatText(value=0.0, min=0, max=10.0, step=0.01, disabled=False, layout={'width': '100px'})

    # create input widget
    inpt_label1 = widgets.Label(value='Koncentration(y): ')
    input_widget1 = widgets.BoundedFloatText(value=0.0, min=0, max=10.0, step=0.01, disabled=False, layout={'width': '100px'})

    regression_label = widgets.Label(value='Hældningskoefficient: ')
    regression_label1 = widgets.Label(value=str("{:.4f}".format(model.coef_[0][0])))

    # create button widget
    button_x_y = widgets.Button(
        description='x -> y',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click to convert input value to prediction'
    )

    # create button widget
    button_y_x = widgets.Button(
        description='y -> x',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click to convert input value to prediction'
    )

    def inverse_prediction(y):
        return (y) / model.coef_[0]

    
    def on_button_clicked(b):
        plt.clf()
        out.clear_output(wait=True)
        # get input value
        input_value = float(input_widget.value)
        predicted_y = model.predict([[input_value]])
        input_widget1.value = "{:.4f}".format(predicted_y[0][0])
        with out:
            # Plot the scatter plot with the linear regression line
            plt.plot(x_range, y_range, color='r')
            plt.scatter(x, y)
            # plot the predicted value as a green dot
            plt.scatter(input_value, predicted_y, color='green')
            predicted_y = predicted_y[0][0]
            # add dashed lines from axis to green dot
            plt.plot([input_value, input_value], [0, predicted_y], linestyle='dashed', color="orange", zorder=0)
            plt.plot([0, input_value], [predicted_y, predicted_y], linestyle='dashed', color="orange", zorder=0)
            # Add labels and title to the plot
            plt.xlabel('Absorbans')
            plt.ylabel('Koncentration i mg')
            plt.title('Linear Regression Model')
            plt.show()

    def on_button_clicked_1(b):
        plt.clf()
        out.clear_output(wait=True)
        # get input value
        input_value = float(input_widget1.value)
        predicted_y = inverse_prediction(input_value)
        input_widget.value = "{:.4f}".format(predicted_y[0])
        with out:
            # Plot the scatter plot with the linear regression line
            plt.plot(x_range, y_range, color='r')
            plt.scatter(x, y)
            # plot the predicted value as a green dot
            plt.scatter(predicted_y, input_value, color='green')
            predicted_y = predicted_y
            plt.plot([predicted_y, predicted_y], [0, input_value], linestyle='dashed', color="orange", zorder=0)
            plt.plot([0, predicted_y], [input_value, input_value], linestyle='dashed', color="orange", zorder=0)
            # Add labels and title to the plot
            plt.xlabel('Absorbans')
            plt.ylabel('Koncentration i mg')
            plt.title('Linear Regression Model')
            plt.show()


    # attach the function to button click event
    button_x_y.on_click(on_button_clicked)
    button_y_x.on_click(on_button_clicked_1)
    grid = widgets.GridspecLayout(6, 8)
    grid[0, 0] = regression_label
    grid[0, 1] = regression_label1
    grid[1, 1] = input_widget
    grid[1, 0] = inpt_label
    grid[2, 1] = input_widget1
    grid[2, 0] = inpt_label1
    grid[4, 0] = button_x_y
    grid[4, 1] = button_y_x

    # display the widgets
    display(out)
    display(grid)
