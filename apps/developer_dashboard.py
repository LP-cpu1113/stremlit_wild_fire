import streamlit as st
import pandas as pd
import numpy as np

FILE1_PATH = "./models/resnet18_training_history.csv"
FILE2_PATH = "./models/mobilenet_v3_small_training_history.csv"

PATHS = [FILE1_PATH, FILE2_PATH]


def get_training_dataframe(path_str):

    training_data = pd.read_csv(path_str)

    st.dataframe(training_data)

    return training_data["architecture"][0], training_data


def select_model(model_names):

    option = st.selectbox(
        "Select the model",
        options=model_names
    )

    st.write(f"You selected {option}!")

    return option


def main():

    st.title("ML Dashboard")
    st.text("This is my first app.")

    a = np.linspace(0, 10, 6)
    st.text(a)

    model_storage= {}
    for path_str in PATHS:

        model_name, model_df = get_training_dataframe(path_str)

        model_storage[model_name]=model_df

    model_names = list(model_storage.keys())
    selected_option = select_model(model_names)


    st.subheader(f" {selected_option}: Loss during training history")
    
    selected_df = model_storage[selected_option]

    # how do I acces the right training history from just name
    #filtered_df = df[df["architecture"] == selected_option]
    st.line_chart(selected_df, x = "epoch", y = ["val_accuracy", "train_loss"])




if __name__ == "__main__":
    main()