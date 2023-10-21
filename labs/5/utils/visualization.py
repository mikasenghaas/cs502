import ipywidgets as widgets
from IPython.display import display

# Display the dropdown
def init():
    # Define the dropdown
    mode_dropdown = widgets.Dropdown(
        options=['DEBUG', 'RUN'],
        value='DEBUG',
        description='Mode:',
    )

    display(mode_dropdown)
    return mode_dropdown