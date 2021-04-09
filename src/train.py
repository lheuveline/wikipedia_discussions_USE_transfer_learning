import pandas as pd
import time
import ast
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf

from skmultilearn.model_selection import iterative_train_test_split

import tensorflow_hub as hub
from tensorflow.keras import layers


# BATCH_SIZE = 256 # Big enough to measure an F1-score
BATCH_SIZE = 1024
AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically to reduce GPU and CPU idle time
SHUFFLE_BUFFER_SIZE = 1024 # Shuffle the training data by a chunck of 1024 observations

EPOCHS = 30


def format_labels(labels_str):
    
    labels_str = labels_str.strip()
    
    # Try to cast to array from csv-string
    try:
        labels = ast.literal_eval(labels_str)
        if isinstance(labels, str):
            labels = [labels]
    # If ValueError, N classes is probably equal to 1 and expressed as "valid" string
    except ValueError:
        labels = [labels_str]
    # If any other error, label can't be understood
    except:
        labels = None
        
    # Check for malformed categories from wikimedia extract
    if isinstance(labels, int):
        labels = None  
    # Check for empty labels
    elif isinstance(labels, list):
        if len(labels) == 0:
            labels = None
        
    return labels

def remove_label(array, to_remove):
    new_arr = []
    for label in array:
        if label.strip() not in to_remove:
            new_arr.append(label)
    return new_arr
            

def filter_labels(array, to_keep):
    new_arr = []
    for label in array:
        if label in to_keep:
            new_arr.append(label)
    if len(new_arr) < 1:
        new_arr = None
        
    return new_arr


def ckeck_errors(df):

    # Check for type  and empty arrays errors
    errors = {}
    for e in df.categories.dropna().values:
        if not isinstance(e, list):
            errors[e] = "datatype: {}".format(type(e))
        if len(e) < 1:
            errors[str(e)] = "array len < 1"
            
    print("N type errors:", len(errors))

    if len(errors) > 0:
        print(errors)


def make_dataset(X, y):

    """
    Prepare dataset from training.
    """

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    print(dataset)

    return dataset



def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost.
    Average (1 - soft-F1) across all labels.
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    
    return macro_cost

def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)
    
    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive
        
    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1


def load_data(filename):

    """
    Load data from filename.
    """

    df = pd.read_csv(
        filename, 
        quotechar='"',
        header=None, 
        usecols=range(2)
    )
    df.columns = ["text", "categories"]
    df.dropna(inplace=True)
    return df

def preprocess_data(df):

    """
    Preprocess data for multilabel classification.
    """

    df.categories = df.categories.apply(format_labels)

    # Remove empty labels introduced by transformation
    df = df.dropna()

    # Downsample to fit in memory for development
    n_samples = 500000
    df = df.sample(n_samples)
    df = df.reset_index(drop=True)

    category_counts = df.explode('categories').value_counts("categories")

    # Manually found outliers in top-100
    # Probably legitimate labels or outliers from Spark
    outliers = [
        "avancement=B", "avancement=AdQ", "WP1.0=oui", 
        "avancement=BA", "avancement=A", "élevée", "maximum",
        "moyenne"
    ]

    # Too frequent
    too_freq = ["Sélection transversale", "Sélection francophone"]
    to_remove = outliers + too_freq

    df["categories"] = df.categories.apply(remove_label, args = (to_remove, ))

    # Filter again after removing labels
    df = df.loc[df.categories.apply(len) > 0]

    # Filter too short texts
    df = df.loc[df.text.apply(len) > 50]

    category_counts = df \
        .explode('categories') \
        .value_counts("categories")

    max_labels = 10
    top_labels = category_counts.head(max_labels)

    df["categories"] = df.categories.apply(filter_labels, args = (top_labels.index,))
    df = df.dropna()

    return df


def compile_model(verbose=True):
    
    """Compile model from Universal Sentence Encoder and add softmax layer for classification"""
    
    embed_layer = hub.KerasLayer(
        "https://tfhub.dev/google/universal-sentence-encoder/4", input_shape=[]
        )
    embed_layer.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(), dtype=tf.string),
        embed_layer,
        layers.Dense(1024, activation='relu', name='hidden_layer'),
        layers.Dense(10, activation='sigmoid', name='output')
    ])

    if verbose:
        model.summary()

    return model

def train(model, train_dataset, test_dataset, lr = 1e-3):
    
    """
    
    Train model
    
    lr : learning rate (original = 1e-5)

    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=macro_soft_f1,
        metrics=[macro_f1]
    )

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset
    )

    return model, history


def main():

    start = time.time()

    input_filename = "../data/frwiki_discussions_categories_processed.csv/part-00000-381f0f76-28b9-4da9-8cb0-96958b5ea46e-c000.csv"
    df = load_data(input_filename)
    print("Load took:", (time.time() - start))

    
    df = preprocess_data(df)

    
    encoder = MultiLabelBinarizer()
    labels_df = pd.DataFrame(encoder.fit_transform(df.categories.values))

    X_train, y_train, X_test, y_test = iterative_train_test_split(
        df.text.values.reshape(-1, 1),
        labels_df.values,
        test_size = 0.5
    )

    train_dataset = make_dataset(X_train, y_train)
    test_dataset = make_dataset(X_test, y_test)

    model = compile_model(verbose = True)
    model, history = train(model, train_dataset, test_dataset, lr = 1e-3)
    
    
    output_dir = "../tf_model/frwikipedia_10_categories_classifier"
    
    model.save(output_dir)

    end = time.time()
    print("Complete training took :", (end - start))

if __name__ == "__main__":
    main()