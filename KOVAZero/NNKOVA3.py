import chess
import chess.pgn
import chess.engine
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.callbacks import TensorBoard
import os
import glob
from tqdm import tqdm
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
OUTPUT_FOLDER_CHESS_DATA = "/Volumes/Lena/Fynn/16MGames/ChessDataPROCESSED"
OUTPUT_FOLDER_RANDOM = "/Volumes/Lena/Fynn/16MGames/RandomEvalPROCESSED"
OUTPUT_FOLDER_TACTIC = "/Volumes/Lena/Fynn/16MGames/TacticDataPROCESSED"
MODEL_FOLDER = "/Volumes/Lena/Fynn/16MGames/ModelWeights"
os.makedirs(MODEL_FOLDER, exist_ok=True)
MODEL_WEIGHTS_PATH = os.path.join(MODEL_FOLDER, "chess_model_weights_zero.weights.h5")
BATCH_SIZE = 256
EPOCHS_PER_CHUNK = 10  # Increased epochs per chunk to give early stopping a chance
LEARNING_RATE = 0.0001
PATIENCE = 5          # Number of epochs with no improvement to wait before stopping
LOG_DIR = os.path.join(MODEL_FOLDER, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


# === UTILITY FUNCTIONS ===
def board_to_tensor(board):
    tensor = np.zeros((8, 8, 14), dtype=np.int8)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row = square // 8
        col = square % 8
        plane = piece.piece_type - 1
        if piece.color == chess.BLACK:
            plane += 6
        tensor[row][col][plane] = 1
    return tensor

def normalize_cp(cp):
    if pd.isna(cp):
        return 0.0
    try:
        cp = int(cp)
        if cp > 1000:
            return 1.0
        elif cp < -1000:
            return -1.0
        return cp / 1000.0
    except ValueError:
        if isinstance(cp, str):
            if cp.startswith('+M') or cp.startswith('#'):
                return 1.0
            elif cp.startswith('-M'):
                return -1.0
            elif cp.startswith('>'):
                try:
                    value = int(cp[1:])
                    return 1.0 if value > 0 else -1.0
                except ValueError:
                    return 0.0
            elif cp.startswith('<'):
                try:
                    value = int(cp[1:])
                    return -1.0 if value < 0 else 1.0
                except ValueError:
                    return 0.0
        return 0.0

# === NEW RESIDUAL BLOCK FUNCTION ===
def residual_block(x, filters):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

# === NEW ALPHA ZERO STYLE MODEL ===
def build_alpha_zero_style_model(policy_output_size=4672, residual_blocks=5):
    input_layer = tf.keras.Input(shape=(8, 8, 14))

    # Initial convolutional block
    x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Residual blocks
    for _ in range(residual_blocks):
        x = residual_block(x, 256)

    # --- Value Head ---
    v = tf.keras.layers.Conv2D(1, (1, 1), padding='same', use_bias=False)(x)
    v = tf.keras.layers.BatchNormalization()(v)
    v = tf.keras.layers.ReLU()(v)
    v = tf.keras.layers.Flatten()(v)
    v = tf.keras.layers.Dense(256, activation='relu')(v)
    v = tf.keras.layers.Dropout(0.5)(v)
    value_output = tf.keras.layers.Dense(1, activation='tanh', name='value_output')(v)

    # --- Policy Head ---
    p = tf.keras.layers.Conv2D(2, (1, 1), padding='same', use_bias=False)(x)
    p = tf.keras.layers.BatchNormalization()(p)
    p = tf.keras.layers.ReLU()(p)
    p = tf.keras.layers.Flatten()(p)
    policy_output = tf.keras.layers.Dense(policy_output_size, activation='softmax', name='policy_output')(p)

    model = tf.keras.Model(inputs=input_layer, outputs=[value_output, policy_output])

    # Compilation is done in main, but this is a good default for the function
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #     loss={'value_output': 'mean_squared_error', 'policy_output': 'categorical_crossentropy'},
    #     metrics={'value_output': 'mae', 'policy_output': 'accuracy'},
    #     loss_weights={'value_output': 1.0, 'policy_output': 1.0}
    # )

    return model

# === TRAINING ===
def train_on_chunk(model, input_files, target_files, epochs=EPOCHS_PER_CHUNK, batch_size=BATCH_SIZE, chunk_index=0, callbacks=None):
    X_chunk_list = []
    y_value_chunk_list = []
    histories = []

    print(f"Loading and training on {len(input_files)} files for up to {epochs} epochs (Chunk {chunk_index})...")
    for input_file, target_file in zip(input_files, target_files):
        print(f"Loading {input_file} and {target_file}")
        try:
            X_chunk_list.append(np.load(input_file))
            y_value_chunk_list.append(np.load(target_file))
        except Exception as e:
            print(f"Error loading file {input_file} or {target_file}: {e}")
            # If an error occurs, it's better to return None or handle the error more gracefully
            # Returning None here means the main loop needs to check for None
            return None, None 

    if X_chunk_list and y_value_chunk_list:
        try:
            X_combined_chunk = np.concatenate(X_chunk_list, axis=0)
            y_value_combined_chunk = np.concatenate(y_value_chunk_list, axis=0)
            dummy_policy_targets = np.zeros((y_value_combined_chunk.shape[0], 4672), dtype=np.float32) # Still using dummy targets

            # Shuffle the combined data
            indices = np.arange(X_combined_chunk.shape[0])
            np.random.shuffle(indices)
            X_shuffled_chunk = X_combined_chunk[indices]
            y_value_shuffled_chunk = y_value_combined_chunk[indices]
            dummy_policy_shuffled_targets = dummy_policy_targets[indices]

            print("Combined chunk X shape:", X_shuffled_chunk.shape)
            print("Combined chunk y_value shape:", y_value_shuffled_chunk.shape)
            print("Dummy policy targets shape:", dummy_policy_shuffled_targets.shape)

            history = model.fit(X_shuffled_chunk,
                                  {'value_output': y_value_shuffled_chunk, 'policy_output': dummy_policy_shuffled_targets},
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  validation_split=0.2,
                                  verbose=1,
                                  callbacks=callbacks)
            histories.append(history)
        except MemoryError as e:
            print(f"Memory error during chunk training: {e}")
            return None, None # Return None on MemoryError
    else:
        print("No data loaded for this chunk.")
        return None, None # Return None if no data is loaded

    return model, histories

def plot_training_history(all_histories, output_path="training_history_zero_early_stopping.png"):
    plt.figure(figsize=(12, 6))
    epochs_total = 0
    all_mae = []
    all_val_mae = []

    for history in all_histories:
        if 'value_output_mae' in history.history and 'val_value_output_mae' in history.history:
            epochs = len(history.history['value_output_mae'])
            epochs_range = range(epochs_total, epochs_total + epochs)
            all_mae.extend(history.history['value_output_mae'])
            all_val_mae.extend(history.history['val_value_output_mae'])
            epochs_total += epochs

    plt.plot(range(epochs_total), all_mae, label='Training MAE')
    plt.plot(range(epochs_total), all_val_mae, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (Value)')
    plt.title('Training and Validation Mean Absolute Error with Early Stopping')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()

# === MAIN ===
def main():
    # === CONFIGURATION ===
    OUTPUT_FOLDER_CHESS_DATA = "/Volumes/Lena/Fynn/16MGames/ChessDataPROCESSED"
    OUTPUT_FOLDER_RANDOM = "/Volumes/Lena/Fynn/16MGames/RandomEvalPROCESSED"
    OUTPUT_FOLDER_TACTIC = "/Volumes/Lena/Fynn/16MGames/TacticDataPROCESSED"
    MODEL_FOLDER = "/Volumes/Lena/Fynn/16MGames/ModelWeights"
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    MODEL_WEIGHTS_PATH_PREFIX = os.path.join(MODEL_FOLDER, "chess_model_weights_chunk") # Prefix for saving weights
    FINAL_MODEL_WEIGHTS_PATH = os.path.join(MODEL_FOLDER, "NNKOVA3_weights_final.weights.h5")
    BATCH_SIZE = 256
    EPOCHS_PER_CHUNK = 10
    LEARNING_RATE = 0.0001
    PATIENCE = 5

    # === TRAINING ===
    all_input_files = []
    all_target_files = []

    # Collect files from chessData
    input_pattern_chess = os.path.join(OUTPUT_FOLDER_CHESS_DATA, "analyzed_inputs_chunk_*.npy")
    target_pattern_chess = os.path.join(OUTPUT_FOLDER_CHESS_DATA, "analyzed_targets_chunk_*.npy")
    all_input_files.extend(sorted(glob.glob(input_pattern_chess)))
    all_target_files.extend(sorted(glob.glob(target_pattern_chess)))

    # Collect files from random_evals
    input_pattern_random = os.path.join(OUTPUT_FOLDER_RANDOM, "analyzed_inputs_chunk_*.npy")
    target_pattern_random = os.path.join(OUTPUT_FOLDER_RANDOM, "analyzed_targets_chunk_*.npy")
    all_input_files.extend(sorted(glob.glob(input_pattern_random)))
    all_target_files.extend(sorted(glob.glob(target_pattern_random)))

    # Collect files from tactic_evals
    input_pattern_tactic = os.path.join(OUTPUT_FOLDER_TACTIC, "analyzed_inputs_chunk_*.npy")
    target_pattern_tactic = os.path.join(OUTPUT_FOLDER_TACTIC, "analyzed_targets_chunk_*.npy")
    all_input_files.extend(sorted(glob.glob(input_pattern_tactic)))
    all_target_files.extend(sorted(glob.glob(target_pattern_tactic)))

    if not all_input_files or not all_target_files or len(all_input_files) != len(all_target_files):
        print(f"Error: Could not find matching input/target files in the specified output folders for training.")
        return

    # Use the new AlphaZero style model
    model = build_alpha_zero_style_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss={'value_output': 'mean_squared_error', 'policy_output': 'categorical_crossentropy'},
                  metrics={'value_output': 'mae', 'policy_output': 'accuracy'},
                  loss_weights={'value_output': 1.0, 'policy_output': 0.0}) # Policy output loss weight is 0.0 as before

    # Load existing weights (if you trained the old model and saved its weights to FINAL_MODEL_WEIGHTS_PATH)
    # Be aware that only compatible layers will load weights.
    if os.path.exists(FINAL_MODEL_WEIGHTS_PATH):
        print(f"Loading existing model weights from {FINAL_MODEL_WEIGHTS_PATH}")
        try:
            model.load_weights(FINAL_MODEL_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
            print("Successfully loaded weights (some layers might have been skipped due to mismatch).")
        except Exception as e:
            print(f"Error loading weights: {e}. Starting training from scratch for the new architecture.")
            # If loading fails, continue training from scratch (newly initialized weights)
    else:
        print("No existing final model weights found. Starting training from scratch.")

    all_training_histories = []
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    early_stopping_callback = callbacks.EarlyStopping(monitor='val_value_output_mae',
                                                      patience=PATIENCE,
                                                      restore_best_weights=True,
                                                      verbose=1,
                                                      mode='min')
    callbacks_list = [early_stopping_callback, tensorboard_callback]
    print(f"Found {len(all_input_files)} data chunks to train on.")
    
    for i in range(len(all_input_files)):
        input_chunk_files = [all_input_files[i]]
        target_chunk_files = [all_target_files[i]]
        print(f"Training on chunk {i+1}/{len(all_input_files)}")
        
        # Pass the callbacks_list to train_on_chunk
        model, history = train_on_chunk(model, input_chunk_files, target_chunk_files,
                                        epochs=EPOCHS_PER_CHUNK, chunk_index=i, callbacks=callbacks_list)
        
        if model is None: # Check if training was interrupted (e.g., MemoryError, file loading error)
            print("Training interrupted due to an error in train_on_chunk. Exiting.")
            return

        if history:
            all_training_histories.extend(history)
            # The early stopping callback object within callbacks_list holds the stopped_epoch attribute
            # We need to check if early stopping was triggered globally or reset within each chunk
            # For per-chunk early stopping, the callback should be recreated or reset per chunk.
            # However, for a single global early stopping across all chunks,
            # using the same callback instance as you do is correct.
            # The .stopped_epoch is only set if early stopping actually triggers.
            if early_stopping_callback.stopped_epoch > 0:
                print(f"Early stopping triggered at epoch {early_stopping_callback.stopped_epoch + 1} (absolute epochs). Stopping overall training.")
                break # Exit the loop if early stopping was triggered

        # Save weights after each chunk with chunk number in filename
        chunk_weights_path = f"{MODEL_WEIGHTS_PATH_PREFIX}_{i+1}.weights.h5"
        model.save_weights(chunk_weights_path)
        print(f"Saved model weights to {chunk_weights_path}")

    # Save final weights
    model.save_weights(FINAL_MODEL_WEIGHTS_PATH)
    print(f"Final model weights saved to {FINAL_MODEL_WEIGHTS_PATH}")

    # Plot the training history
    plot_training_history(all_training_histories, os.path.join(MODEL_FOLDER, "training_history_zero_intermediate_save.png"))

if __name__ == "__main__":
    main()