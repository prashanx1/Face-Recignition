# monitor_and_enroll_with_blacklist.py

import face_recognition
import cv2
import os
import time

# --- Configuration ---
# 1. The folder with all the new images arriving.
source_images_dir = r"D:\face-based duplicate detection\images"

# 2. The folder where your permanent face database is stored.
database_dir = r"D:\face-based duplicate detection\stored-faces"

# 3. The file to keep track of images we have already processed.
PROCESSED_LOG_FILE = "processed_log.txt"

# --- NEW CONFIGURATION ---
# 4. The folder containing images of blacklisted individuals.
BLACKLIST_DIR = r"D:\face-based duplicate detection\black_listed"

# 5. How strict the comparison is.
MATCH_TOLERANCE = 0.55

# 6. How much padding to add for new faces.
PADDING = 30

# --- Setup ---
os.makedirs(database_dir, exist_ok=True)
os.makedirs(BLACKLIST_DIR, exist_ok=True) # Ensure blacklist folder exists


# --- Function to load a list of already processed files ---
def load_processed_files(log_file):
    """Reads the log file and returns a set of filenames."""
    if not os.path.exists(log_file):
        return set()
    with open(log_file, 'r') as f:
        return set(line.strip() for line in f)

# --- Function to load face encodings from a directory ---
def load_face_database(directory, name):
    """Loads all face encodings from a given directory."""
    encodings = []
    filenames = []
    print(f"Loading {name} database...")
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            filepath = os.path.join(directory, filename)
            try:
                image = face_recognition.load_image_file(filepath)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    encodings.append(face_encodings[0])
                    filenames.append(filename)
            except Exception as e:
                print(f"Warning: Could not load or encode {filename} from {name}. Error: {e}")
    print(f"{name.capitalize()} database loaded with {len(encodings)} faces.")
    return encodings, filenames


# --- Main Loop ---
if __name__ == "__main__":
    
    processed_files = load_processed_files(PROCESSED_LOG_FILE)
    print(f"Found {len(processed_files)} previously processed images.")
    
    # Load both the main database AND the new blacklist database
    known_encodings, known_names = load_face_database(database_dir, "main")
    blacklist_encodings, blacklist_names = load_face_database(BLACKLIST_DIR, "blacklist")

    print("\nStarting to monitor the images folder for new files...")
    
    while True:
        all_source_files = {f for f in os.listdir(source_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
        new_files_to_process = all_source_files - processed_files
        
        if not new_files_to_process:
            print("No new images to process. Waiting...")
            time.sleep(10)
            continue

        print(f"\nFound {len(new_files_to_process)} new image(s) to process.")

        for new_filename in new_files_to_process:
            new_filepath = os.path.join(source_images_dir, new_filename)
            print(f"  -> Processing: {new_filename}")

            try:
                new_image = face_recognition.load_image_file(new_filepath)
                face_locations = face_recognition.face_locations(new_image, model='hog')
                face_encodings = face_recognition.face_encodings(new_image, face_locations)
            except Exception as e:
                print(f"    -> Error loading image: {e}")
                # Log the file as processed even if it's broken
                with open(PROCESSED_LOG_FILE, 'a') as f: f.write(new_filename + '\n')
                processed_files.add(new_filename)
                continue

            if not face_encodings:
                print("    -> Warning: No face found. Marking as processed and skipping.")
                with open(PROCESSED_LOG_FILE, 'a') as f: f.write(new_filename + '\n')
                processed_files.add(new_filename)
                continue

            new_encoding = face_encodings[0]

            # --- Blacklist Check (Priority 1) ---
            is_blacklisted = False
            if blacklist_encodings:
                matches = face_recognition.compare_faces(blacklist_encodings, new_encoding, tolerance=MATCH_TOLERANCE)
                if True in matches:
                    is_blacklisted = True
                    first_match_index = matches.index(True)
                    matched_name = blacklist_names[first_match_index]
                    print(f"    -> !!! BLACKLIST ALERT !!! Person matches blacklisted file '{matched_name}'.")
            
            # --- If person is NOT blacklisted, proceed with normal logic ---
            if not is_blacklisted:
                # --- Duplicate Check (Priority 2) ---
                match_found = False
                if known_encodings:
                    matches = face_recognition.compare_faces(known_encodings, new_encoding, tolerance=MATCH_TOLERANCE)
                    if True in matches:
                        print("    -> DUPLICATE DETECTED in main database. Not adding.")
                        match_found = True
                
                # --- If not a duplicate, add to database ---
                if not match_found:
                    print("    -> New person detected. Adding to database...")
                    top, right, bottom, left = face_locations[0]
                    top, right, bottom, left = max(0, top - PADDING), min(new_image.shape[1], right + PADDING), min(new_image.shape[0], bottom + PADDING), max(0, left - PADDING)
                    face_crop = new_image[top:bottom, left:right]
                    face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                    database_filepath = os.path.join(database_dir, new_filename)
                    cv2.imwrite(database_filepath, face_crop_bgr)
                    
                    # Update our "in-memory" database
                    known_encodings.append(new_encoding)
                    known_names.append(new_filename)
                    print(f"    -> Success! Saved as '{new_filename}'.")

            # --- Log this file as processed (regardless of outcome) ---
            with open(PROCESSED_LOG_FILE, 'a') as f:
                f.write(new_filename + '\n')
            processed_files.add(new_filename)
            print(f"    -> Marked '{new_filename}' as processed.")