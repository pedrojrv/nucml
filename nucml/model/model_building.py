import os
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import wandb
from wandb.keras import WandbCallback



def get_callbacks(name, logs_dir_name="logs", lr_method="plateau", reset=False, patience_epochs=10, save_freq=7531*5, append_wandb=False):
    logdir = logs_dir_name
    chkpoint_dir = os.path.abspath(os.path.join(logs_dir_name, "checkpoints"))
    csv_logger_dir = os.path.join(logs_dir_name, "training_metrics.csv")

    if not os.path.exists(chkpoint_dir):
        os.makedirs(chkpoint_dir)

    chkpoint_dir = os.path.abspath(os.path.join(logs_dir_name, "checkpoints/best_model.hdf5"))

    if lr_method == "plateau":
        all_callbacks =  [
            tfdocs.modeling.EpochDots(),
            tf.keras.callbacks.CSVLogger(csv_logger_dir),
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_epochs, restore_best_weights=True), # val_loss
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
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_epochs, restore_best_weights=True), # val_loss
            tf.keras.callbacks.ModelCheckpoint(chkpoint_dir, monitor='loss', verbose=0,
                                               save_best_only=True, save_weights_only=False,
                                               save_frequency=save_freq),
            tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1, profile_batch=0)]
    else:
        all_callbacks = [
            tfdocs.modeling.EpochDots(),
            tf.keras.callbacks.CSVLogger(csv_logger_dir),
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience_epochs, restore_best_weights=True), # val_loss
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
                steps_per_epoch = STEPS_PER_EPOCH,
                epochs=max_epochs,
                validation_data=(x_test, y_test),
                callbacks=get_callbacks(name, logs_dir_name=logs_dir_name, lr_method=lr_method, append_wandb=append_wandb),
                verbose=verbose, 
                initial_epoch=initial_epoch)   
        return history
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            steps_per_epoch = STEPS_PER_EPOCH,
            epochs=max_epochs,
            validation_data=(x_test, y_test),
            callbacks=get_callbacks(name, logs_dir_name=logs_dir_name, lr_method=lr_method, append_wandb=append_wandb),
            verbose=verbose, 
            initial_epoch=initial_epoch)   
        return history


def compile_and_fit_lw(model, name, x_train, y_train, x_test, y_test, BATCH_SIZE=120, 
                    max_epochs=5, DECAY_EPOCHS=10, lr_method="plateau", initial_epoch=0):
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
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=(x_test, y_test),
        callbacks=get_callbacks(name, lr_method=lr_method),
        verbose=1, 
        initial_epoch=initial_epoch)   
    return history

















# def tf_dataset_gen(x, y, xt, yt, BUFFER_SIZE, BATCH_SIZE, N_TRAIN, gpu=False, multiplier=2, cache=False):
#     if gpu == True:
#         BATCH_SIZE = BATCH_SIZE * multiplier
#         print("GPU: ON")
#     train_dataset = tf.data.Dataset.from_tensor_slices((x.values, y.values)).shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)
#     test_dataset = tf.data.Dataset.from_tensor_slices((xt.values, yt.values)).batch(BATCH_SIZE)
#     if cache == True: # Ensures loader doesnt re-read data at each epoch.
#         train_dataset = train_dataset.cache()
#         test_dataset = test_dataset.cache()
#     STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE
#     print("BATCH SIZE: ", BATCH_SIZE)
#     print("STEPS PER EPOCH: ", STEPS_PER_EPOCH)
#     return train_dataset, test_dataset, STEPS_PER_EPOCH, BATCH_SIZE


# def get_optimizer(lr_schedule):
#     return tf.keras.optimizers.Adam(lr_schedule)


# def compile_and_fit(model, name, train_dataset, test_dataset, STEPS_PER_EPOCH, BATCH_SIZE=None, 
#                     optimizer=None, max_epochs=10000, DECAY_EPOCHS=1000):
#     lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#         0.001, decay_steps=STEPS_PER_EPOCH*DECAY_EPOCHS,
#         decay_rate=1, staircase=False)
    
#     if optimizer is None:
#         optimizer = get_optimizer(lr_schedule)
#     model.compile(optimizer=optimizer,
#                   loss='mse',
#                   metrics=['mae', 'mse'])
#     model.summary()
#     history = model.fit(
#         train_dataset,
#         steps_per_epoch = STEPS_PER_EPOCH,
#         epochs=max_epochs,
#         validation_data=test_dataset,
#         callbacks=get_callbacks(name),
#         verbose=0)        
#     return history


# def train_ensdf_network(strategy):
# 	with strategy.scope():
# 	    tiny_model = tf.keras.Sequential([
# 	        layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
# 	        layers.Dense(1)])
# 	    history = compile_and_fit(tiny_model, 'sizes/Tiny_CPU', train_dataset, 
# 	                                                 test_dataset, STEPS_PER_EPOCH_CPU, max_epochs=5000)
# 	    return history



# N_VALIDATION = len(x_test)
# N_TRAIN = len(x_train)
# FEATURES = len(x_train.columns)
# BUFFER_SIZE = N_TRAIN
# BATCH_SIZE_PER_REPLICA = 64
# BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


# train_dataset, test_dataset, STEPS_PER_EPOCH_CPU, BATCH_SIZE_CPU = tf_dataset_gen(
#     x_train, y_train, x_test, y_test, BUFFER_SIZE, BATCH_SIZE)



# # def compile_and_fit(model, name, train_dataset, test_dataset, STEPS_PER_EPOCH, BATCH_SIZE=None, 
# #                     optimizer=None, max_epochs=10000, DECAY_EPOCHS=1000):
# def compile_and_fit(model, name, x_train, y_train, x_test, y_test, STEPS_PER_EPOCH, BATCH_SIZE=None, 
#                     optimizer=None, max_epochs=10000, DECAY_EPOCHS=1000):
#     lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#         0.001, decay_steps=STEPS_PER_EPOCH*DECAY_EPOCHS,
#         decay_rate=1, staircase=False)
    
#     if optimizer is None:
#         optimizer = get_optimizer(lr_schedule)
#     model.compile(optimizer=optimizer,
#                   loss='mse',
#                   metrics=['mae', 'mse'])
#     model.summary()
# #     history = model.fit(
# #         train_dataset,
# #         steps_per_epoch = STEPS_PER_EPOCH,
# #         epochs=max_epochs,
# #         validation_data=test_dataset,
# #         callbacks=get_callbacks(name),
# #         verbose=1)   
#     history = model.fit(
#         x_train, y_train,
#         batch_size=BATCH_SIZE,
#         steps_per_epoch = STEPS_PER_EPOCH,
#         epochs=max_epochs,
#         validation_data=(x_test, y_test),
#         callbacks=get_callbacks(name),
#         verbose=1)   
#     return history
