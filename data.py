import json
import os
import numpy as np

def load_spotify_data(data_folder_name="data", max_playlists=5000):
    """Parses the official Spotify JSON format into raw text playlists."""
    raw_playlists = []
    
    # Resolve the exact physical path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    secure_data_path = os.path.join(base_dir, data_folder_name)
    print(f"Checking physical disk at: {secure_data_path}")
    
    # Check if the folder physically exists
    if not os.path.exists(secure_data_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Windows cannot see the 'data' folder at all.")
        
    # Read EVERYTHING in the folder (Bypassing glob)
    all_files = os.listdir(secure_data_path)
    print(f"Found {len(all_files)} total items in the folder.")
    
    # Manually filter for JSON files
    json_files = [f for f in all_files if f.endswith('.json')]
    
    if not json_files:
        print(f"WHAT PYTHON SEES IN THE FOLDER: {all_files[:10]}")
        raise FileNotFoundError("The folder exists, but Python sees NO files ending in '.json'. (Is OneDrive hiding them?)")
        
    json_files.sort()  # deterministic order across runs
    print(f"Found {len(json_files)} JSON files. Loading until {max_playlists} playlists collected...")

    for file_name in json_files:
        file_path = os.path.join(secure_data_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for playlist in data['playlists']:
            tracks = [track['track_name'] for track in playlist['tracks']]
            if len(tracks) > 3:
                raw_playlists.append(tracks)

        if len(raw_playlists) >= max_playlists:
            break

    raw_playlists = raw_playlists[:max_playlists]
    print(f"Successfully loaded {len(raw_playlists)} valid playlists.")
    return raw_playlists

# Tokenize: Convert each song (a string/URI) into an integer ID. Assign each unique song a number to map song -> ID
# Slice: Cut each playlist into fixed-size training examples of the form (input window -> next song), a playlist is a sequence,
# slide a window across it and for each window input X is those songs, target Y is the song that comes right after
# raw playlists (song names)
#    │ tokenize: song → integer ID  (build vocab)
#    ▼
# tokenized playlists ([10, 11, 12, 13, 14])
#    │ slide a window of context_length
#    ▼
# X = [10,11,12]  Y = 13     ← each is one training example
#    │
#    ▼
# X feeds model.forward (embedding → blocks → last pos → matchmaker → logits)
# Y becomes the one-hot target in cross_entropy_loss
def tokenize_and_slice(raw_playlists, context_length=3, test_split=0.1, min_freq=2):
    """
    Builds the vocabulary (song ↔ ID maps) over all data.
    Splits playlists into train/test sets.
    Tokenizes + slices each set into (X, Y) integer arrays.
    Returns the train/test arrays, vocab_size, and the ID→song map (for turning predictions back into song names later).
    """
    print("\n--- TOKENIZATION & SPLITTING ---")

    # Count how often each song appears across all playlists, songs that appear < min_freq times are dropped
    # Songs that don't appear often enough can't help the model learn anything (think learning a word by reading it in 1 sentence/context vs 100 sentences/contexts)
    # Over multiple runs, the model only knows to push that song in the very few contexts it appears in (think learning a word by reading it in 1 sentence/context 30 times)
    counts = {}
    for playlist in raw_playlists:
        for track in playlist:
            counts[track] = counts.get(track, 0) + 1
    
    # Build Dictionary on ALL data to prevent KeyErrors, this is the Tokenize step
    # Build the vocab from ONLY songs seen >= min_freq times
    track_to_id = {}
    id_to_track = {}
    current_id = 0
    for playlist in raw_playlists:
        for track in playlist:
            if counts[track] >= min_freq and track not in track_to_id:
                track_to_id[track] = current_id
                id_to_track[current_id] = track
                current_id += 1
                
    vocab_size = len(track_to_id) # Num. of unqiue songs/catalog size
    print(f"Vocabulary built! Kept {vocab_size} songs (>= {min_freq} plays); "
          f"dropped {len(counts) - vocab_size} rare songs.")
    
    # Shuffle and split the raw playlists into train/test at the playlist level
    # 90/10 split, train model on first 90% of playlists, test model on last 10% of playlists
    np.random.shuffle(raw_playlists)
    split_idx = int(len(raw_playlists) * (1 - test_split))
    train_raw = raw_playlists[:split_idx]
    test_raw = raw_playlists[split_idx:]
    
    # Helper function to slice a specific subset
    def process_subset(subset):
        X_data, Y_data = [], []
        for playlist in subset:
            # Map each song to its ID, or None if it was filtered out (out-of-vocab)
            tokenized = [track_to_id.get(track, None) for track in playlist]
            # For each position i, X_data gets tokenized[i: i+context_length] - a window of context_length consecutive songs (the INPUT)
            # Y_data gets tokenized [i + context_length] - the very next song after that window (the TARGET)
            for i in range(len(tokenized) - context_length):
                window = tokenized[i : i + context_length]
                target = tokenized[i + context_length]
                # Skip any window touching a rare song (input OR target). Alternative is to delete song outright or replace it with UNK signal
                # Deleting it is feeding bad data to the model (A, UNK, B) -> (A, B) implies A then B which is false. Replacing with UNK signal
                # would lead to the model attending to UNK (which is awkward) or predicting UNK (which is awkward)
                # We skip windows to keep adjacency truthful and every kept example fully learnable.
                if target is None or None in window:
                    continue
                X_data.append(window)
                Y_data.append(target)
        # X_data is what flows through the embedding -> transformer, Y_data only appears as the one-hot target in cross_entropy_loss
        return np.array(X_data), np.array(Y_data)

    # Process the sets independently
    X_train, Y_train = process_subset(train_raw)
    X_test, Y_test = process_subset(test_raw)
    
    print(f"Train Sequences: {len(X_train)} | Test (Vault) Sequences: {len(X_test)}")
    # vocab_size = the num. of unique songs in the catalog, id_to_track maps song IDs to the actual song name
    return X_train, Y_train, X_test, Y_test, vocab_size, id_to_track