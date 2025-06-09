import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.callbacks import TensorBoard
import os
import glob
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from selfplay import generate_self_play_data # Your new self_play module
# You'll also need the move encoding/decoding, maybe put it in utilities or mcts.py
from NNKOVA3 import board_to_tensor, EPOCHS_PER_CHUNK, BATCH_SIZE, LEARNING_RATE, PATIENCE, LOG_DIR, MODEL_FOLDER, build_alpha_zero_style_model, plot_training_history  
import pandas as pd
# === CONFIGURATION ===
# ... (existing config) ...
SELF_PLAY_DATA_FOLDER = "/Volumes/Lena/Fynn/16MGames/SelfPlayData"
os.makedirs(SELF_PLAY_DATA_FOLDER, exist_ok=True)
NUM_SELF_PLAY_GAMES_PER_ITERATION = 100 # Number of games to play before training
NUM_MCTS_SIMULATIONS = 800 # Number of MCTS simulations per move (AlphaZero typically uses ~800)
# ... (rest of config) ...


# ... (board_to_tensor, normalize_cp, residual_block, build_alpha_zero_style_model - unchanged) ...

# MODIFIED train_on_chunk function to accept policy targets
def train_on_chunk(model, input_files, value_target_files, policy_target_files, epochs=EPOCHS_PER_CHUNK, batch_size=BATCH_SIZE, chunk_index=0, callbacks=None, val_data=None):
    X_chunk_list = []
    y_value_chunk_list = []
    y_policy_chunk_list = [] # NEW: Policy targets list
    histories = []

    print(f"Loading and training on {len(input_files)} files for up to {epochs} epochs (Chunk {chunk_index})...")
    for input_file, value_target_file, policy_target_file in zip(input_files, value_target_files, policy_target_files):
        print(f"Loading {input_file}, {value_target_file}, and {policy_target_file}")
        try:
            X_chunk_list.append(np.load(input_file))
            y_value_chunk_list.append(np.load(value_target_file))
            y_policy_chunk_list.append(np.load(policy_target_file)) # NEW: Load policy targets
        except Exception as e:
            print(f"Error loading file {input_file} or its target files: {e}")
            return None, None

    if X_chunk_list and y_value_chunk_list and y_policy_chunk_list:
        try:
            X_combined_chunk = np.concatenate(X_chunk_list, axis=0)
            y_value_combined_chunk = np.concatenate(y_value_chunk_list, axis=0)
            y_policy_combined_chunk = np.concatenate(y_policy_chunk_list, axis=0) # NEW: Concatenate policy targets

            # Shuffle the combined data
            indices = np.arange(X_combined_chunk.shape[0])
            np.random.shuffle(indices)
            X_shuffled_chunk = X_combined_chunk[indices]
            y_value_shuffled_chunk = y_value_combined_chunk[indices]
            y_policy_shuffled_chunk = y_policy_combined_chunk[indices] # NEW: Shuffle policy targets

            print("Combined chunk X shape:", X_shuffled_chunk.shape)
            print("Combined chunk y_value shape:", y_value_shuffled_chunk.shape)
            print("Combined chunk y_policy shape:", y_policy_shuffled_chunk.shape) # NEW: Print policy shape

            history = model.fit(X_shuffled_chunk,
                                  {'value_output': y_value_shuffled_chunk, 'policy_output': y_policy_shuffled_chunk}, # NEW: Use real policy targets
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  validation_data=val_data,
                                  verbose=1,
                                  callbacks=callbacks)
            histories.append(history)
        except MemoryError as e:
            print(f"Memory error during chunk training: {e}")
            return None, None
    else:
        print("No data loaded for this chunk.")
        return None, None

    return model, histories


# ... (plot_training_history - unchanged) ...


# === MAIN ===
def main():
    # ... (existing CONFIGURATION inside main - you can remove if using global config) ...
    # Collect files from all sources
    all_input_files = []
    all_value_target_files = []
    # MODIFICATION: Add lists for policy target files
    all_policy_target_files = []


    # Placeholder for a self-play loop
    # In a real AlphaZero, this loop would generate data and then train.
    # We'll start with generating data for one iteration, then integrate it.

    # Build the model
    model = build_alpha_zero_style_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss={'value_output': 'mean_squared_error', 'policy_output': 'categorical_crossentropy'},
                  metrics={'value_output': 'mae', 'policy_output': 'accuracy'},
                  loss_weights={'value_output': 1.0, 'policy_output': 1.0}) # Both loss weights are now 1.0

    # Load existing weights (important for continuous learning)
    # Be aware that only compatible layers will load weights.
    # The 'NNKOVA3_weights_final.weights.h5' contains the dual-head model structure.
    # If starting fresh or with old single-head weights, you might need to handle load_weights error.
    FINAL_MODEL_WEIGHTS_PATH = os.path.join(MODEL_FOLDER, "NNKOVA3_weights_final.weights.h5")
    if os.path.exists(FINAL_MODEL_WEIGHTS_PATH):
        print(f"Loading existing model weights from {FINAL_MODEL_WEIGHTS_PATH}")
        try:
            # use by_name and skip_mismatch when loading weights from potentially different architecture versions
            model.load_weights(FINAL_MODEL_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
            print("Successfully loaded weights (some layers might have been skipped due to mismatch).")
        except Exception as e:
            print(f"Error loading weights: {e}. Starting training from scratch for the new architecture.")
    else:
        print("No existing final model weights found. Starting training from scratch.")


    # --- AlphaZero Iteration Loop (Conceptual) ---
    # This is the outer loop that makes it truly AlphaZero-like
    # For now, we'll just demonstrate one data generation and training step.
    num_iterations = 1 # Or more if you want multiple self-play -> train cycles

    for iteration in range(num_iterations):
        print(f"\n--- AlphaZero Iteration {iteration + 1} ---")

        # Step 1: Generate Self-Play Data
        print("Generating self-play data...")
        # Note: generate_self_play_data needs to be able to call your model's .predict()
        # and has the responsibilty of mapping policy output to actual moves.
        # Ensure 'model' is passed to it correctly and that 'board_to_tensor'
        # and 'decode_policy_output' are accessible/implemented in self_play.py
        input_data_path, policy_data_path, value_data_path = generate_self_play_data(
            model=model,
            num_games=NUM_SELF_PLAY_GAMES_PER_ITERATION,
            output_dir=SELF_PLAY_DATA_FOLDER,
            num_simulations=NUM_MCTS_SIMULATIONS
        )

        # Step 2: Add newly generated data to training sets
        all_input_files = [input_data_path] # For this iteration, use newly generated data
        all_value_target_files = [value_data_path]
        all_policy_target_files = [policy_data_path]

        # MODIFICATION: Separate a static validation set (adjust num_validation_files if needed)
        # For simplicity in this self-play loop, we'll use a small part of the NEW data for validation.
        # In a very large scale AZ system, you might have a dedicated validation set from historical data.
        num_validation_samples_per_file = int(np.load(all_input_files[0]).shape[0] * 0.1) # 10% of new data for validation
        
        # Load the newly generated data
        X_all_iter = np.load(all_input_files[0])
        y_value_all_iter = np.load(all_value_target_files[0])
        y_policy_all_iter = np.load(all_policy_target_files[0])

        # Create static validation set from this iteration's data
        X_val_static = X_all_iter[:num_validation_samples_per_file]
        y_val_static = y_value_all_iter[:num_validation_samples_per_file]
        y_policy_val_static = y_policy_all_iter[:num_validation_samples_per_file]

        # Use the rest for training
        X_train_iter = X_all_iter[num_validation_samples_per_file:]
        y_value_train_iter = y_value_all_iter[num_validation_samples_per_file:]
        y_policy_train_iter = y_policy_all_iter[num_validation_samples_per_file:]

        val_data = (X_val_static, {'value_output': y_val_static, 'policy_output': y_policy_val_static})
        print(f"Static validation set for iteration {iteration+1} has {X_val_static.shape[0]} samples.")


        # Step 3: Train the Neural Network
        all_training_histories = [] # Reset histories for each iteration for clarity
        tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
        early_stopping_callback = callbacks.EarlyStopping(monitor='val_value_output_mae',
                                                          patience=PATIENCE,
                                                          restore_best_weights=True,
                                                          verbose=1,
                                                          mode='min')
        callbacks_list = [early_stopping_callback, tensorboard_callback]
        
        # For this example, we'll treat the loaded iteration data as one big chunk
        print(f"Training model on {X_train_iter.shape[0]} samples for iteration {iteration+1}...")
        history = model.fit(X_train_iter,
                              {'value_output': y_value_train_iter, 'policy_output': y_policy_train_iter},
                              epochs=EPOCHS_PER_CHUNK, # You might want more epochs per iteration in a real AZ
                              batch_size=BATCH_SIZE,
                              validation_data=val_data,
                              verbose=1,
                              callbacks=callbacks_list)
        
        if history:
            all_training_histories.append(history)
            if early_stopping_callback.stopped_epoch > 0:
                print(f"Early stopping triggered in iteration {iteration+1}. Stopping overall training.")
                break


        # Save model weights after each iteration (adjust naming as needed)
        iteration_weights_path = os.path.join(MODEL_FOLDER, f"NNKOVA4_iter_{iteration+1}.weights.h5")
        model.save_weights(iteration_weights_path)
        print(f"Saved model weights for iteration {iteration+1} to {iteration_weights_path}")

    # Save final weights
    model.save_weights(FINAL_MODEL_WEIGHTS_PATH)
    print(f"Final model weights saved to {FINAL_MODEL_WEIGHTS_PATH}")

    # Plot the training history (this will only plot the last iteration's history in this setup)
    # You might want to collect histories across iterations for a cumulative plot.
    plot_training_history(all_training_histories, os.path.join(MODEL_FOLDER, "training_history_NNKOVA4_final.png"))

if __name__ == "__main__":
    main()