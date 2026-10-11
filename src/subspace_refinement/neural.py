"""Validated neural refiner with the upstream 2×128 ReLU architecture.

Architecture adapted from Dennis Noll's weight_refiner (MIT); training loop
and event-level holdout protocol are supplied by this study. See notices.
TensorFlow is imported only when a neural comparison is requested.
"""
import numpy as np


def tensorflow_setup():
    import tensorflow as tf
    try:
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
    except RuntimeError:
        pass  # A caller may already have initialized its TensorFlow runtime.
    return tf


def fit_network(data, seed, projection=None):
    tf = tensorflow_setup()
    x, xv = data.train_x, data.valid_x
    if projection is not None:
        x, xv = x@projection, xv@projection
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(x.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                  loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    optimizer = model.optimizer

    @tf.function(input_signature=[tf.TensorSpec([None, x.shape[1]], tf.float32),
                                 tf.TensorSpec([None, 1], tf.float32),
                                 tf.TensorSpec([None], tf.float32)])
    def step(a, labels, weights):
        with tf.GradientTape() as tape:
            prediction = model(a, training=True)
            loss = loss_fn(labels, prediction, sample_weight=weights)
        optimizer.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))

    x, xv = x.astype('float32'), xv.astype('float32')
    y = (data.train_w > 0).astype('float32')[:, None]
    yv = (data.valid_w > 0).astype('float32')[:, None]
    weights, wv = np.abs(data.train_w).astype('float32'), np.abs(data.valid_w).astype('float32')
    rng = np.random.default_rng(seed)
    best_loss, best_weights, selected_epoch, stale = np.inf, None, 0, 0
    for epoch in range(1, 101):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 128):
            ids = order[start:start + 128]
            step(x[ids], y[ids], weights[ids])
        loss = float(loss_fn(yv, model(xv, training=False), sample_weight=wv))
        if loss < best_loss:
            best_loss, best_weights, selected_epoch, stale = loss, model.get_weights(), epoch, 0
        else:
            stale += 1
        if stale >= 20:
            break
    model.set_weights(best_weights)
    return model, {'selected_epoch': selected_epoch, 'epochs_run': epoch,
                   'best_validation_bce': best_loss}


def predict_refinement(model, x, projection=None):
    z = x if projection is None else x@projection
    prediction = np.concatenate([np.asarray(model(a.astype('float32'), training=False))[:, 0]
                                 for a in np.array_split(z, max(1, len(z)//5000))])
    return 2*prediction - 1

