import os
import numpy as np
import cv2
import tensorflow as tf
import csv
from keras.applications import ResNet50
from keras.models import Model
from collections import Counter

def load_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

def extract_features_with_resnet50(model, img_path):
    img = preprocess_image(img_path)
    features = model.predict(img)
    feature_vector = features.flatten()
    median_value = np.median(feature_vector)
    binary_vector = (feature_vector > median_value).astype(int)
    return binary_vector

def process_fingerprint_images(root_folder, output_file):
    all_features = []
    labels = []
    model = load_resnet50_model()

    for user_folder in os.listdir(root_folder):
        user_path = os.path.join(root_folder, user_folder)
        if os.path.isdir(user_path):
            has_tif = False
            for file_name in os.listdir(user_path):
                if file_name.endswith('.tif'):
                    has_tif = True
                    file_path = os.path.join(user_path, file_name)
                    print(f"Processing {file_path}...")
                    feature_vector = extract_features_with_resnet50(model, file_path)
                    print(f"Feature vector for {file_name}: {feature_vector[:10]}...") 
                    all_features.append(feature_vector)
                    labels.append(user_folder)
            if not has_tif:
                print(f"Skipping folder {user_folder} (no .tif images)")

    np.savez(output_file, features=all_features, labels=labels)
    print(f"Saved features to {output_file}")

def predict_user(test_image_path, feature_file):
    data = np.load(feature_file)
    saved_features = data['features']
    saved_labels = data['labels']

    model = load_resnet50_model()

    test_feature_vector = extract_features_with_resnet50(model, test_image_path)

    min_distance = float('inf')
    predicted_label = None
    for saved_feature, label in zip(saved_features, saved_labels):
        distance = np.linalg.norm(test_feature_vector - saved_feature)
        if distance < min_distance:
            min_distance = distance
            predicted_label = label

    score = 1 - (min_distance / np.linalg.norm(test_feature_vector))
    return predicted_label, score

def calculate_metrics(predictions, actual_labels):
    correct = sum([1 for p, a in zip(predictions, actual_labels) if p == a])
    accuracy = correct / len(actual_labels)

    frr = sum([1 for p, a in zip(predictions, actual_labels) if p != a and a == actual_labels[0]]) / len(actual_labels)
    far = sum([1 for p, a in zip(predictions, actual_labels) if p != a and p == actual_labels[0]]) / len(predictions)

    return accuracy, frr, far

def main():
    mode = input("Choose mode (1: Process .tif images, 2: Predict user): ")

    root_folder = "output_dataset/fingerprint"
    output_file = "fingerprint_features.npz"
    test_folder = "output_dataset/test"

    if mode == '1':
        process_fingerprint_images(root_folder, output_file)

    elif mode == '2':
        predictions = []
        actual_labels = []
        scores = []

        if not os.path.exists(output_file):
            print("Feature file not found, please process the images first.")
            return

        for test_subfolder in os.listdir(test_folder):
            test_subfolder_path = os.path.join(test_folder, test_subfolder)
            if os.path.isdir(test_subfolder_path):
                subfolder_predictions = []
                subfolder_scores = []
                
                for file_name in os.listdir(test_subfolder_path):
                    if file_name.endswith('.tif'):
                        file_path = os.path.join(test_subfolder_path, file_name)
                        print(f"Predicting for {file_path}...")

                        predicted_label, score = predict_user(file_path, output_file)

                        subfolder_predictions.append(predicted_label)
                        subfolder_scores.append(score)

                if subfolder_predictions:
                    combined_label = Counter(subfolder_predictions).most_common(1)[0][0]
                    combined_score = np.mean(subfolder_scores)  
                else:
                    combined_label, combined_score = None, None

                predictions.append(combined_label)
                actual_labels.append(test_subfolder)  
                scores.append(combined_score)

        accuracy, frr, far = calculate_metrics(predictions, actual_labels)

        
        with open('results_finger.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Test Folder", "Predicted Label", "Score"])

            for test_subfolder in os.listdir(test_folder):
                test_subfolder_path = os.path.join(test_folder, test_subfolder)
                if os.path.isdir(test_subfolder_path):
                    writer.writerow([test_subfolder, combined_label, combined_score])

            writer.writerow([])
            writer.writerow(["Metrics", "Value"])
            writer.writerow(["Accuracy", f"{accuracy:.4f}"])
            writer.writerow(["False Rejection Rate (FRR)", f"{frr:.4f}"])
            writer.writerow(["False Acceptance Rate (FAR)", f"{far:.4f}"])

        print(f"Accuracy: {accuracy:.4f}")
        print(f"False Rejection Rate (FRR): {frr:.4f}")
        print(f"False Acceptance Rate (FAR): {far:.4f}")
        print("Results saved to results_finger.csv")

    else:
        print("Invalid mode selected. Please choose 1 or 2.")

if __name__ == "__main__":
    main()
