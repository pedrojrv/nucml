"""Model building utilities."""
import os
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
from wandb.keras import WandbCallback


def get_callbacks(name, logs_dir_name="logs", lr_method="plateau", patience_epochs=10, save_freq=7531*5,
                  append_wandb=False):
    """Get a callback list for TensorFlow training.

    Args:
        name (str): Name of the model.
        logs_dir_name (str, optional): Name of the directory where the callback logs will be stored. Defaults to "logs".
        lr_method (str, optional): Learning rate method to implement. Defaults to "plateau".
        patience_epochs (int, optional): Number of epochs to wait before stopping training due to lack of validation
            progress. Defaults to 10.
        save_freq (int, optional): Determines how many steps are needed to create a checkpoint. Defaults to 7531*5.
        append_wandb (bool, optional): If True, the WANDB callback will be added. Defaults to False.

    Returns:
        list: List containing all TensorFlow callbacks.
    """
    logdir = logs_dir_name
    chkpoint_dir = os.path.abspath(os.path.join(logs_dir_name, "checkpoints"))
    csv_logger_dir = os.path.join(logs_dir_name, "training_metrics.csv")

    if not os.path.exists(chkpoint_dir):
        os.makedirs(chkpoint_dir)

    chkpoint_dir = os.path.abspath(os.path.join(logs_dir_name, "checkpoints/best_model.hdf5"))

    if lr_method == "plateau":
        all_callbacks = [
            tfdocs.modeling.EpochDots(),
            tf.keras.callbacks.CSVLogger(csv_logger_dir),
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_epochs, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(chkpoint_dir, monitor='loss', verbose=0,
                                               save_best_only=True, save_weights_only=False,
                                               save_frequency=save_freq),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.8, patience=4, verbose=1, mode='auto',
                min_delta=0.0001, cooldown=2, min_lr=0.001),
            tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=0)]
    elif lr_method == "normal":
        all_callbacks = [
            tfdocs.modeling.EpochDots(),
            tf.keras.callbacks.CSVLogger(csv_logger_dir),
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_epochs, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(chkpoint_dir, monitor='loss', verbose=0,
                                               save_best_only=True, save_weights_only=False,
                                               save_frequency=save_freq),
            tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=0)]
    else:
        all_callbacks = [
            tfdocs.modeling.EpochDots(),
            tf.keras.callbacks.CSVLogger(csv_logger_dir),
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_epochs, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(chkpoint_dir, monitor='loss', verbose=0,
                                               save_best_only=True, save_weights_only=False,
                                               save_frequency=save_freq),
            tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=0)]
    if append_wandb:
        all_callbacks.append(WandbCallback())
    return all_callbacks


def compile_and_fit(model, name, x_train, y_train, x_test, y_test, BATCH_SIZE=120,
                    max_epochs=5, DECAY_EPOCHS=10, lr_method="plateau", initial_epoch=0,
                    logs_dir_name="logs", append_wandb=False, verbose=0, comet=False, comet_exp=None):
    """Compile and fit a TensorFlow model.

    Args:
        model (object): TensorFlow model object.
        name (str): Name by which the TensorFlow model will be saved.
        x_train (np.array): The X-train numpy array set.
        y_train (np.array): The y-train numpy array.
        x_test (np.array): The X-test numpy array set.
        y_test (np.array): The y-test numpy array.
        BATCH_SIZE (int, optional): Batch size for the tensorflow dataset. Defaults to 120.
        max_epochs (int, optional): Max number of epochs to train for. Defaults to 5.
        DECAY_EPOCHS (int, optional): Number of epochs before slightly decreasing the learning rate. Defaults to 10.
        lr_method (str, optional): Type of learning rate adjustment method. Defaults to "plateau".
        initial_epoch (int, optional): Initial epoch of the provided model. Defaults to 0.
        logs_dir_name (str, optional): Name of the directory where the logs will be stored. Defaults to "logs".
        append_wandb (bool, optional): If True, the WANDB callback will be appended. Defaults to False.
        verbose (int, optional): See the TensorFlow verbosity for more information. Defaults to 0.
        comet (bool, optional): If True, the training will be under a Comet experiment. Defaults to False.
        comet_exp (object, optional): The Comet experiment by which the training will happen. Defaults to None.

    Returns:
        object: Training history object.
    """
    STEPS_PER_EPOCH = len(x_train) // BATCH_SIZE

    if lr_method == "plateau" or lr_method == "normal":
        model.compile(optimizer=tf.keras.optimizers.Adam(0.005), loss='mse', metrics=['mae', 'mse'])
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                0.005, decay_steps=STEPS_PER_EPOCH*DECAY_EPOCHS, decay_rate=0.5)),
            loss='mse',
            metrics=['mae', 'mse'])
    if comet:
        with comet_exp.train():
            history = model.fit(
                x_train, y_train,
                batch_size=BATCH_SIZE,
                steps_per_epoch=STEPS_PER_EPOCH,
                epochs=max_epochs,
                validation_data=(x_test, y_test),
                callbacks=get_callbacks(
                    name, logs_dir_name=logs_dir_name, lr_method=lr_method, append_wandb=append_wandb),
                verbose=verbose,
                initial_epoch=initial_epoch)
        return history
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=max_epochs,
            validation_data=(x_test, y_test),
            callbacks=get_callbacks(name, logs_dir_name=logs_dir_name, lr_method=lr_method, append_wandb=append_wandb),
            verbose=verbose,
            initial_epoch=initial_epoch)
        return history


def compile_and_fit_lw(model, name, x_train, y_train, x_test, y_test, BATCH_SIZE=120,
                       max_epochs=5, DECAY_EPOCHS=10, lr_method="plateau", initial_epoch=0):
    """Compile and fit a TensorFlow model.

    Args:
        model (object): TensorFlow model to fit.
        name (str): Name by which the model will be saved.
        x_train (np.array): The X-train numpy array set.
        y_train (np.array): The y-train numpy array.
        x_test (np.array): The X-test numpy array set.
        y_test (np.array): The y-test numpy array.
        BATCH_SIZE (int, optional): Batch size for the tensorflow dataset. Defaults to 120.
        max_epochs (int, optional): Max number of epochs to train for. Defaults to 5.
        DECAY_EPOCHS (int, optional): Number of epochs before slightly decreasing the learning rate. Defaults to 10.
        lr_method (str, optional): Type of learning rate adjustment method. Defaults to "plateau".
        initial_epoch (int, optional): Initial epoch of the provided model. Defaults to 0.

    Returns:
        object: Training history object.
    """
    STEPS_PER_EPOCH = len(x_train) // BATCH_SIZE

    if lr_method == "plateau" or lr_method == "normal":
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse', metrics=['mae', 'mse'])
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                0.001, decay_steps=STEPS_PER_EPOCH*DECAY_EPOCHS, decay_rate=1)),
            loss='mse',
            metrics=['mae', 'mse'])
    # model.summary()

    model.load_weights(os.path.join("./model_checkpoints/"))
    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=(x_test, y_test),
        callbacks=get_callbacks(name, lr_method=lr_method),
        verbose=1,
        initial_epoch=initial_epoch)
    return history


def get_xgboost_params(eta=0.5, gamma=0, l2=0, max_depth=30, grow_policy='depthwise', max_bin=256, determ_hist='true',
                       objective="rmse", resume=False, gpu_id=0):
    """Get a dictionary containing xgboost parameters."""
    objectives = {'rmse': 'reg:squarederror', 'huber': 'reg:pseudohubererror', 'rmsle': 'reg:squaredlogerror'}
    new_or_resume = {True: 'update', False: 'default'}
    # specify parameters via map
    params = {
        # GENERAL PARAMETERS
        "booster": "gbtree",
        "verbosity": 1,  # 0 (silent), 1 (warning), 2 (info), 3 (debug)
        # 'nthread':4,

        # TREE BOOSTER PARAMETERS
        'eta': eta,  # learning rate: 0.3
        'max_depth': max_depth,  # max depth: 6, 0 is lossguided avaliable for gpu_hist and hist
        "tree_method": "gpu_hist",
        'grow_policy': grow_policy,  # controls new nodes: 'depthwise', 'lossguide' avaliable
        'max_leaves': 0,  # when lossguide, max number of nodes to be added: 0
        'max_bin': max_bin,  # max num of discrete bins for continues features: 256
        'num_parallel_tree': 1,  # num parallel trees constructed during each iteration: 1
        "process_type": new_or_resume[resume],

        "gamma": gamma,  # min loss reduction for partition: 0 - REGULARIZE UP
        'min_child_weight': 1,  # min sum of instance weights for child: 1 - REGULARIZE UP
        "lambda": l2,  # l2 reg:1 - REGULARIZE UP
        'alpha': 0,  # l1 reg:0 - REGULARIZE UP


        # GPU HIST PARAMETER
        'deterministic_histogram': determ_hist,  # pre-rounding routing leads to lower accuracy: true
        'gpu_id': gpu_id,

        # LEARNING TASK PARAMETERS
        'objective': objectives[objective],  # squarederror, pseudohubererror, squaredlogerror
        'eval_metric': ['rmse', 'mae'],  # eval metric for validation data
    }

    return params


def tf_dataset_gen(x, y, xt, yt, BUFFER_SIZE, BATCH_SIZE, N_TRAIN, gpu=False, multiplier=2, cache=False):
    """Get a tensorflow dataset generator."""
    if gpu:
        BATCH_SIZE = BATCH_SIZE * multiplier
        print("GPU: ON")
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x.values, y.values)).shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((xt.values, yt.values)).batch(BATCH_SIZE)
    if cache:  # Ensures loader doesnt re-read data at each epoch.
        train_dataset = train_dataset.cache()
        test_dataset = test_dataset.cache()
    STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
    print("BATCH SIZE: ", BATCH_SIZE)
    print("STEPS PER EPOCH: ", STEPS_PER_EPOCH)
    return train_dataset, test_dataset, STEPS_PER_EPOCH, BATCH_SIZE


def get_optimizer(lr_schedule):
    """Return adamn optimizer given a learning rate schedule."""
    return tf.keras.optimizers.Adam(lr_schedule)
