import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.decomposition import PCA

def create_model(label_dict):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(label_dict), activation='softmax') 
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def prepare_face_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = []
    label_dict = {}
    label = 0

    for dir_name in dirs:
        if not dir_name.startswith("."):
            subject_dir_path = os.path.join(data_folder_path, dir_name)
            subject_images_names = os.listdir(subject_dir_path)

            for image_name in subject_images_names:
                if not image_name.startswith("."):
                    image_path = os.path.join(subject_dir_path, image_name)
                    image = cv2.imread(image_path)
                    try:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    except:
                        print('ghh')

                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

                    for (x, y, w, h) in faces_rects:
                        face = gray[y:y+w, x:x+h]
                        face = cv2.resize(face, (128, 128)) 
                        faces.append(face)
                        labels.append(label)
            label_dict[label] = dir_name
            label += 1

    return np.array(faces), np.array(labels), label_dict


def train_and_save_model(faces, labels, label_dict, model_file):
    print("Training the model...")
    faces = faces / 255.0
    faces = faces.reshape(-1, 128, 128, 1)

    model = create_model(label_dict)
    model.fit(faces, labels, epochs=10, batch_size=32)
    model.save(model_file)
    print(f"Model saved to {model_file}")
    return model


def extract_face_features(face_images, model, pca_model=None):
    face_images = face_images / 255.0
    face_images = face_images.reshape(-1, 128, 128, 1)

    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = feature_extractor.predict(face_images)

    if pca_model:
        features = pca_model.transform(features) 

    binary_features = (features > 0).astype(np.uint8)  
    print(binary_features)

    return binary_features


def process_face_dataset(data_folder_path, model, pca_model=None, output_file="face_features.npz"):
    faces, labels, label_dict = prepare_face_data(data_folder_path)
    print(f"Prepared {len(faces)} face images for feature extraction.")

    user_labels = [f"user_{label+1}" for label in labels] 

    features = extract_face_features(faces, model, pca_model)
    print(features)
    print(f"Extracted features shape: {features.shape}")
    print(user_labels)
    np.savez(output_file, features=features, labels=user_labels)
    print(f"Saved face features and labels to {output_file}")

    return label_dict


def main():
    model_file = 'face_cnn_model_f.h5'
    data_folder_path = "output_dataset/face"

    faces, labels, label_dict = prepare_face_data(data_folder_path)

    if os.path.exists(model_file):
        print(f"Loading model from {model_file}...")
        model = tf.keras.models.load_model(model_file)
    else:
        model = train_and_save_model(faces, labels, label_dict, model_file)

    apply_pca = True
    pca_model = None
    if apply_pca:
        pca_model = PCA(n_components=30)
        faces = faces / 255.0
        faces = faces.reshape(-1, 128, 128, 1)
        feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
        features = feature_extractor.predict(faces)
       

        n_samples, n_features = features.shape
        n_components = min(n_samples, n_features, 30)  
        print(f"Shape of features: {features.shape}")
        print(f"Using n_components={n_components} for PCA")

        pca_model = PCA(n_components=n_components)
        pca_model.fit(features)


    process_face_dataset(data_folder_path, model, pca_model)


if __name__ == "__main__":
    main()
