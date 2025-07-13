from model.randomforest import RandomForest
from Config import Config
import pandas as pd
import numpy as np
from modelling.data_model import Data
from sklearn.preprocessing import LabelEncoder

def model_predict(data, df, name):
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

def model_predict_chained(X, df, name):
    results = {}

    # STEP 1: Predict y2 (Intent)
    df['y'] = df[Config.TYPE_COLS[0]]  # y2
    data_y2 = Data(X, df)

    if data_y2.X_train is None:
        print(f"Skipping group {name} — not enough valid y2 records.")
        return

    print(f"[{name}] Training model for y2 (Intent)")
    model_y2 = RandomForest("RandomForest_y2", data_y2.get_embeddings(), data_y2.get_type())
    model_y2.train(data_y2)
    pred_y2 = model_y2.mdl.predict(data_y2.X_test)
    results['y2'] = pred_y2
    print("[y2] Classification Report:")
    model_y2.predictions = pred_y2
    model_y2.print_results(data_y2)

    # Save train/test indices
    train_idx = data_y2.train_idx
    test_idx = data_y2.test_idx

    # STEP 2: Predict y3 (Tone), using y2 as input
    df['y'] = df[Config.TYPE_COLS[1]]  # y3

    # Encode y2_train and y2_pred for stacking
    le_y2 = LabelEncoder()
    y2_train_encoded = le_y2.fit_transform(data_y2.y_train.ravel())
    y2_test_encoded = le_y2.transform(pred_y2)

    # Now build data_y3
    data_y3 = Data(X, df, train_idx=train_idx, test_idx=test_idx)
    if data_y3.X_train is None:
        print(f"Skipping group {name} — not enough valid y3 records.")
        return

    # Ensure correct length match for stacking
    y2_train_filtered = y2_train_encoded[:len(data_y3.X_train)]

    # Stack y2 with original embeddings
    X_train_y3 = np.hstack((data_y3.X_train, y2_train_filtered.reshape(-1, 1)))
    X_test_y3 = np.hstack((data_y3.X_test, y2_test_encoded.reshape(-1, 1)))

    data_y3.X_train = X_train_y3
    data_y3.X_test = X_test_y3

    print(f"[{name}] Training model for y3 (Tone)")
    model_y3 = RandomForest("RandomForest_y3", data_y3.X_train, data_y3.get_type())
    model_y3.train(data_y3)
    pred_y3 = model_y3.mdl.predict(data_y3.X_test)
    results['y3'] = pred_y3
    print("[y3] Classification Report:")
    model_y3.predictions = pred_y3
    model_y3.print_results(data_y3)

    # STEP 3: Predict y4 (Resolution Type) using y2 + y3
    df['y'] = df[Config.TYPE_COLS[2]]  # y4

    # Encode y3_train and y3_pred
    le_y3 = LabelEncoder()
    y3_train_encoded = le_y3.fit_transform(data_y3.y_train.ravel())
    y3_test_encoded = le_y3.transform(pred_y3)

    data_y4 = Data(X, df, train_idx=train_idx, test_idx=test_idx)
    if data_y4.X_train is None:
        print(f"Skipping group {name} — not enough valid y4 records.")
        return

    y2_train_filtered = y2_train_encoded[:len(data_y4.X_train)]
    y3_train_filtered = y3_train_encoded[:len(data_y4.X_train)]

    # Stack y2 + y3 with original embeddings
    X_train_y4 = np.hstack((
        data_y4.X_train,
        y2_train_filtered.reshape(-1, 1),
        y3_train_filtered.reshape(-1, 1)
    ))
    X_test_y4 = np.hstack((
        data_y4.X_test,
        y2_test_encoded.reshape(-1, 1),
        y3_test_encoded.reshape(-1, 1)
    ))

    data_y4.X_train = X_train_y4
    data_y4.X_test = X_test_y4

    print(f"[{name}] Training model for y4 (Resolution Type)")
    model_y4 = RandomForest("RandomForest_y4", data_y4.X_train, data_y4.get_type())
    model_y4.train(data_y4)
    pred_y4 = model_y4.mdl.predict(data_y4.X_test)
    results['y4'] = pred_y4
    print("[y4] Classification Report:")
    model_y4.predictions = pred_y4
    model_y4.print_results(data_y4)

    return results

def model_evaluate(model, data):
    model.print_results(data)
