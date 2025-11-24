# smart_database_builder.py

import face_recognition
import cv2
import os

# --- Configuration ---
# 1. The folder with all the original, full-size images.
source_images_dir = r"D:\face-based duplicate detection\images"

# 2. The folder where the clean, cropped face database will be stored.
database_dir = r"D:\face-based duplicate detection\stored-faces"

# 3. How strict the comparison is. Lower is stricter.
MATCH_TOLERANCE = 0.55

# 4. How much padding to add around a face before resizing.
PADDING = 50 
# 5. The desired aspect ratio (width / height) for the final crop. A passport is taller than it is wide.
#    A common ratio is around 4x5 or 3x4. Let's use 4/5 = 0.8
ASPECT_RATIO = 0.80

# --- Setup ---

os.makedirs(database_dir, exist_ok=True)
print("System starting. Building a clean, duplicate-free database.")


# --- Main Logic ---
# This list will hold the face encodings of people we have already added in this session.
known_encodings_in_session = []

# Loop through every file in the source images folder
for filename in os.listdir(source_images_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        
        filepath = os.path.join(source_images_dir, filename)
        print(f"\nProcessing image: {filename}...")

        # Load the image and find faces
        image = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(image, model='hog')
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Check if we found exactly one face
        if len(face_encodings) != 1:
            print(f"  -> Warning: Found {len(face_encodings)} faces in this image. Skipping.")
            continue
        
        new_encoding = face_encodings[0]

        # --- Duplicate Check ---
        match_found = False
        if known_encodings_in_session:
            matches = face_recognition.compare_faces(known_encodings_in_session, new_encoding, tolerance=MATCH_TOLERANCE)
            if True in matches:
                print(f"  -> DUPLICATE DETECTED. This person is already in the database. Skipping.")
                match_found = True

        # --- If NOT a duplicate, process and save the new face ---
        if not match_found:
            print(f"  -> New person found. Cropping and saving...")
            
            # Add this new person to our list for future checks in this session
            known_encodings_in_session.append(new_encoding)
            
            # Get face coordinates
            top, right, bottom, left = face_locations[0]
            
            # Add padding
            top = max(0, top - PADDING)
            right = min(image.shape[1], right + PADDING)
            bottom = min(image.shape[0], bottom + PADDING)
            left = max(0, left - PADDING)
            
            # --- NEW: Enforce Passport-Style Aspect Ratio ---
            crop_height = bottom - top
            crop_width = right - left
            current_ratio = crop_width / crop_height

            if current_ratio > ASPECT_RATIO:
                # Crop is too wide, need to make it taller
                new_height = crop_width / ASPECT_RATIO
                delta = (new_height - crop_height) / 2
                top = max(0, top - delta)
                bottom = min(image.shape[0], bottom + delta)
            else:
                # Crop is too tall, need to make it wider
                new_width = crop_height * ASPECT_RATIO
                delta = (new_width - crop_width) / 2
                left = max(0, left - delta)
                right = min(image.shape[1], right + delta)
            
            # Crop the face with the new, corrected coordinates
            face_crop = image[int(top):int(bottom), int(left):int(right)]
            
            # Convert to BGR for saving
            face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            
            # Save using the original filename for easy reference
            database_filepath = os.path.join(database_dir, filename)
            cv2.imwrite(database_filepath, face_crop_bgr)
            
            print(f"  -> Success! Saved new person to {database_filepath}")

print("\n--------------------")
print("Database creation complete.")