import streamlit as st
import numpy as np
import pandas as pd

FILE1_PATH="./models/resnet18_training_history.csv"

def get_df():
    training_data_1=pd.read_csv(FILE1_PATH)
    st.dataframe(training_data_1)



"""
import streamlit as st
import pandas as pd
from .utils import get_suffix_paths

def get_training_history_df(dir_models="./models/", suffix_csv_file="_training_history.csv"):
    training_history_paths = get_suffix_paths(dir_path=dir_models, suffix=suffix_csv_file)

    history_columns = [
        "architecture",
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    ]

    if training_history_paths:
        # Each CSV contains additional rows, so concatenate rather than merge.
        history_dataframes = [
            pd.read_csv(path)
            for path in training_history_paths
        ]

        df = (
            pd.concat(history_dataframes, ignore_index=True)
            .drop_duplicates(subset=history_columns)
            .sort_values(["architecture", "epoch"])
        )

        if df.empty:
            st.info("No training history data to display.")
            return pd.DataFrame(columns=history_columns)
        else:
            return df
    else:
        st.info("No training history files found.")
        return pd.DataFrame(columns=history_columns)



def show_training_history(dir_models, suffix_csv_file):
    df =  get_training_history_df(dir_models=dir_models, suffix_csv_file=suffix_csv_file)   
    if not df.empty:
            loss_df = df.melt(
                id_vars=["architecture", "epoch"],
                value_vars="train_loss",
                var_name="dataset",
                value_name="loss",
            )

            loss_df["series"] = (
                loss_df["architecture"]
                + " — "
                + loss_df["dataset"].str.replace("_loss", "", regex=False)
            )

            st.line_chart(
                data=loss_df,
                x="epoch",
                y="loss",
                color="series",
                width="stretch",
            )



"""