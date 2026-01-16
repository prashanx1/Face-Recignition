# Face Recognition System 🎯

This is a Python-based Face Recognition project that detects and recognizes human faces from images using computer vision techniques. The project uses pre-stored images of known individuals and compares them with test images to identify matches.

## 🚀 Features
- Face detection from images
- Face recognition using known images
- Accurate matching using facial encodings
- Simple and beginner-friendly implementation
- No webcam required

## 🛠️ Tech Stack
- Python
- OpenCV
- face_recognition
- NumPy

## 📂 Project Structure
Face-Recignition/
├── known_faces/ # Images of known people
├── test_images/ # Images to be tested
├── face_recognition.py
├── requirements.txt
└── README.md

## ⚙️ Installation
1. Clone the repository
git clone https://github.com/prashanx1/Face-Recignition.git

cd Face-Recignition


2. Install dependencies
pip install -r requirements.txt


## ▶️ How to Run
1. Add clear images of known people inside the `known_faces` folder  
2. Add images you want to test inside the `test_images` folder  
3. Run the script

"python face_recognition.py"

4. The program will process the images and display the recognized faces

## 🧠 How It Works
- Loads images from the `known_faces` directory
- Extracts facial features and generates encodings
- Compares test image encodings with known encodings
- Identifies the person if a match is found, otherwise labels as **Unknown**

## 📌 Applications
- Identity verification
- Criminal or suspect identification
- Photo tagging systems
- Security analysis

## 🔮 Future Enhancements
- Real-time webcam recognition
- GUI-based interface
- Database integration
- Improved accuracy using deep learning models

## 👨‍💻 Author
**Prashant Paliwal**  
GitHub: https://github.com/prashanx1
