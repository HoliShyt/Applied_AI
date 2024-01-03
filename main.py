from datasets import AVAILABLE_DATASETS
from models import AVAILABLE_MODELS
from classifier_train_and_test import full_classifier_procedure

from typing import Sequence, Any, Union


# Print all elements of a list formatted such that there's one element per line, with a number in front
# Useful for displaying choices for a multi-choice input
def formatted_numbered_list(list_to_print: Sequence[Any],
                            initial_index: int = 1,
                            nb_format_string: str = "[{}]\t") -> str:
    formatted_list = '\n'.join(list(
        f"{nb_format_string.format(initial_index+i)}{list_to_print[i]}"
        for i in range(len(list_to_print))
    ))
    return formatted_list


# Give the user a multi-choice input among the choices list.
# The user can select an option from its associated number or by using the aliases, which are the option names
# actually displayed on the choice menu.
def user_input_choice(choices: Sequence[Any],
                      choice_prompt: str = None,
                      choice_aliases: Sequence[str] = None,
                      input_prompt: str = ">>> ") -> Any:
    initial_index = 1
    # If not given, aliases is the list of all choices converted to strings
    if not choice_aliases:
        choice_aliases = list(str(choice) for choice in choices)
    else:
        if len(choices) != len(choice_aliases):
            raise ValueError("List of choices and list of choice name aliases must be of same length!")

    # Display prompt and choices
    if choice_prompt:
        print(choice_prompt)
    print(formatted_numbered_list(choice_aliases, initial_index))

    while True:
        # Get user input, check that it's either an alias, or a valid number. Keep looping if it isn't.
        user_input = input(input_prompt)
        if user_input in choice_aliases:
            choice_idx = choice_aliases.index(user_input)
            return choices[choice_idx]
        try:
            user_input_int = int(user_input)
            if user_input_int < initial_index or user_input_int > len(choices):
                print(f"Invalid answer id! Must be between {initial_index} and {len(choices)}.")
            else:
                return choices[user_input_int - initial_index]
        except ValueError:
            print("Invalid answer! Must be one of the choices or its corresponding id number.")


# Give the user a numeric input which can be a float or an integer between two bounds
def user_input_number(prompt: str = "",
                      lower_bound: float = -float('inf'),
                      upper_bound: float = float('inf'),
                      must_be_integer: bool = False,
                      input_prompt: str = ">>> ") -> Union[float, int]:

    if lower_bound > upper_bound:
        raise ValueError(f"Invalid bounds, upper bound ({upper_bound}) must be larger than lower bound ({lower_bound}).")

    # Display prompt
    if prompt:
        print(prompt)

    # Loop while input isn't valid
    while True:
        # Get user input, check that it is parsable as a float
        user_input = input(input_prompt)
        try:
            user_input_float = float(user_input)
            # Then check if it is between the two bounds
            if user_input_float < lower_bound:
                print(f"Invalid input! Must be larger than {lower_bound}.")
            else:
                if user_input_float > upper_bound:
                    print(f"Invalid input! Must be smaller than {upper_bound}.")
                else:
                    # If it doesn't need to be an integer, we can return it
                    if not must_be_integer:
                        return user_input_float

                    try:
                        # Else we have to check that input is parsable as integer as well and return that
                        user_input_int = int(user_input)
                        return user_input_int
                    except ValueError:
                        print("Invalid input! Must be an integer.")
        except ValueError:
            print(f"Invalid input! Must be a valid parsable {'integer' if must_be_integer else 'float'}.")


# MAIN PROGRAM
if __name__ == '__main__':
    print(f"{'='*30}\n\nAPPLIED AI MINIPROJECT\nBy Florian LACHAUX\nGroup 16\n\n{'='*30}\n")
    print("This project aims to train and test models on classification datasets.")

    # Select model type (due to time constraints only MLP was implemented)
    model_type = user_input_choice(AVAILABLE_MODELS, "\nSelect classifier model type:")

    # Select dataset
    dataset_name = user_input_choice(AVAILABLE_DATASETS, "\nSelect dataset to train on:")
    # Input/output size for both implemented datasets is the same, this code would have to change
    # if we added more datasets
    if dataset_name in ('mnist', 'emnist_digits'):
        input_size = 28*28
        output_size = 10
    else:
        raise NotImplementedError()

    # Letting the user select a portion of training data to set apart for validation
    valsplit_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    valsplit_aliases = list(f"{v*100:.0f} %" for v in valsplit_values)
    trainval_split = 1 - user_input_choice(valsplit_values, "\nWhat percentage of training data to set apart for validation?", valsplit_aliases)

    # Choosing the number of epochs
    n_epochs = user_input_number("\nTrain on how many epochs?", 1, must_be_integer=True)

    # CREATING ALL SETS OF PARAMETERS TO TEST
    if model_type in ('mlp',):
        # For MLP, the user can set:
        # - the number and size of all hidden layers
        # - the activation function to put in-between the layers
        print(f"\nTime to make the sets of parameters you want to test!")
        from models import MLPClassifierActs

        parameter_sets = []
        model_no = 1
        done_with_models = False
        while not done_with_models:
            print(f"\nMODEL {model_no}")

            # 1ST STEP : SETTING THE HIDDEN LAYERS
            layers_done = False
            layers = []
            while not layers_done:
                print(f"Current hidden layers: {layers}")
                layer_size = user_input_number("Enter the size of the next hidden layer (or 0 if you're done)",
                                               0, must_be_integer=True)
                if layer_size != 0:
                    layers.append(layer_size)
                else:
                    # Ask for confirmation when the user is done with setting the hidden layers one by one
                    print(f"Your model {model_no} will have the following layer sizes:\n"
                          f"{[input_size]+layers+[output_size]}")
                    layers_done_confirm = user_input_choice((True, False),
                                                            choice_prompt="Confirm?",
                                                            choice_aliases=["Yes", "No"])
                    if layers_done_confirm:
                        layers_done = True

            # 2ND STEP : SETTING THE ACTIVATION FUNCTION
            act_fcn = user_input_choice(list(MLPClassifierActs.keys()), "Select your network's activation function:")

            # STEP 3 : CONFIRMING THE ADDITION OF CURRENT MODEL TO THE SET OF MODELS TO TEST
            print(f"\nModel {model_no}:\n"
                  f"Layers: {[input_size]+layers+[output_size]}\n"
                  f"Activation function: {act_fcn}")

            model_confirm = user_input_choice((True, False),
                                              choice_prompt="Confirm?",
                                              choice_aliases=["Yes", "No"])

            if not model_confirm:
                print("Resetting choices for current model...\n\n")
            else:
                # 4TH STEP : ASK USER IF THEY WANT TO STOP MAKING MODELS OR NOT
                parameter_sets.append([input_size, output_size, layers, act_fcn])
                print(f"Model {model_no} confirmed and added!\n")

                continue_making_models = user_input_choice((True, False),
                                                           choice_prompt="Continue making models?",
                                                           choice_aliases=["Yes", "No"])

                if continue_making_models:
                    model_no += 1
                else:
                    done_with_models = True

        print(f"ALL MODELS DONE! Here's the complete parameter sets:\n{parameter_sets}")
    else:
        raise NotImplementedError()

    # Once all models to test have been made, we can run the full selection and testing process
    input("\nReady!\nPress Enter to launch the model selection process.\n")
    full_classifier_procedure(dataset_name, trainval_split, model_type, n_epochs, parameter_sets)
