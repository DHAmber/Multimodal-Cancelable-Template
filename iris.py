import os
import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates

def extract_iris_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    localized, center, radius = localize_iris(img)
    normalized = segment_and_normalize_iris(localized, center, radius)

    gabor_kernel = log_gabor_filter()
    features = convolve2d(normalized, gabor_kernel, mode="same")
    # cv2.imwrite("localized.png", localized)
    # cv2.imwrite("normalized.png", normalized * 255)
    # cv2.imwrite("filtered.png", features * 255)
    binary_features = (features > 0).astype(np.uint8).flatten()

    return binary_features

def localize_iris(img):
    blurred = cv2.GaussianBlur(img, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=100, param2=30, minRadius=20, maxRadius=80)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0, 0]  
        mask = np.zeros_like(img)
        cv2.circle(mask, (x, y), r, 255, -1)
        localized = cv2.bitwise_and(img, img, mask=mask)
        return localized, (x, y), r
    raise ValueError("Iris could not be localized.")

def segment_and_normalize_iris(localized, center, radius):
    x_center, y_center = center
    height, width = localized.shape
    theta = np.linspace(0, 2 * np.pi, 512)
    r = np.linspace(0, radius, 64)

    rr, tt = np.meshgrid(r, theta, indexing='ij')
    x = rr * np.cos(tt) + x_center
    y = rr * np.sin(tt) + y_center

    normalized = map_coordinates(localized, [y, x], order=1, mode='nearest')
    return normalized

def log_gabor_filter():
    kernel_size = 31
    sigma = kernel_size / 6
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    gabor = np.exp(-(x**2 + y**2) / (2 * sigma**2)) * np.cos(2 * np.pi * x / sigma)
    return gabor

def process_iris_dataset(root_folder, output_file):
    all_features = []
    labels = []
    print("Preprocessing of iris images")
    for user_folder in os.listdir(root_folder):
        user_path = os.path.join(root_folder, user_folder)
        if os.path.isdir(user_path):
            user_features = []
            for image_name in os.listdir(user_path):
                image_path = os.path.join(user_path, image_name)
                if image_path.endswith('.bmp'):
                    try:
                        features = extract_iris_features(image_path)
                        print(features)
                        user_features.append(features)
                    except ValueError as e:
                        print(f"Error processing {image_path}: {e}")
            
            if user_features:
                consolidated_features = np.mean(user_features, axis=0)
                all_features.append(consolidated_features)
                labels.append(user_folder)

    np.savez(output_file, features=all_features, labels=labels)
    print(f"Saved all iris features to {output_file}")

def main():
    iris_root = "output_dataset/iris"
    output_file = "iris_features.npz"
    process_iris_dataset(iris_root, output_file)
    data = np.load("iris_features.npz")
    print(data["features"].shape)
    print(data["labels"])


if __name__ == "__main__":
    main()
