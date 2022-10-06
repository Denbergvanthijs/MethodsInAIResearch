from dotenv import dotenv_values

from dialogstate import DialogState

configurability = dotenv_values(".env")  # Load configuration from .env file
dialog_state = DialogState(configurability=configurability)  # Create DialogState object

# Run the dialog state machine in the command line
while True:
    dialog_state.act()
