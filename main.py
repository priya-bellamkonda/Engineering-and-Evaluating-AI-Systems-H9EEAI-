from preprocess import *
from embeddings import *
from modelling.modelling import model_predict_chained
from Config import Config
import pandas as pd
import numpy as np
import random

# Set random seed
seed = 0
random.seed(seed)
np.random.seed(seed)

def load_data():
    # Load the input data
    df = get_input_data()
    return df

def preprocess_data(df):
    df = de_duplication(df)
    df = noise_remover(df)
    return df

def get_embeddings(df: pd.DataFrame):
    X = get_tfidf_embd(df)  # TF-IDF vector representation
    return X, df

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)

    # Ensure string type for embedding inputs
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype(str)
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype(str)

    grouped_df = df.groupby(Config.GROUPED)

    for name, group_df in grouped_df:
        print(name)
        X, group_df = get_embeddings(group_df)
        model_predict_chained(X, group_df, name)
