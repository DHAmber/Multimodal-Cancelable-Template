import os
import numpy as np
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout,Lambda
import hashlib
import time
from iris import extract_iris_features
from finger import extract_features_with_resnet50
from keras.applications import ResNet50
from keras.models import Model
import utility
import pandas as pd


tz=64
template_size=256

iris_data = np.load("iris_features.npz")
fingerprint_data = np.load("fingerprint_features.npz")
face_data = np.load("face_features.npz")
iris_features = iris_data['features']
iris_labels = iris_data['labels']
fingerprint_features = fingerprint_data['features']
fingerprint_labels = fingerprint_data['labels']
face_features = face_data['features']
face_labels = face_data['labels']

userKey = np.load('bs_list.npy',allow_pickle=True)
test_root_folder = "D:\\Amber\\Multimodel Root\\CancelableTemplate_girls\\output_dataset\\test"

threshold=0.7

header=['Template','Query','Score','Expected Result','Actual Result']

def group_features_by_user(features, labels):
    grouped_features = []
    grouped_labels = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        user_indices = [i for i, l in enumerate(labels) if l == label]
        user_features = features[user_indices]
        averaged_feature = np.mean(user_features, axis=0)
        averaged_feature =(averaged_feature >= 0.5).astype(int)
        grouped_features.append(averaged_feature)
        grouped_labels.append(label)

    return np.array(grouped_features), grouped_labels


iris_features, iris_labels = group_features_by_user(iris_features, iris_labels)
fingerprint_features, fingerprint_labels = group_features_by_user(fingerprint_features, fingerprint_labels)
face_features, face_labels = group_features_by_user(face_features, face_labels)

def fuse_features(iris_features, fingerprint_features, face_features):
    try:
        #iris_features = normalize(iris_features, axis=1)
        #fingerprint_features = normalize(fingerprint_features, axis=1)
        #face_features = normalize(face_features, axis=1)
        fused_features = np.concatenate([face_features,iris_features, fingerprint_features], axis=1)
        return fused_features
    except Exception as e:
        print(e)
        return []

def generate_user_seed_matrix(seed, feature_length):
    dynamic_seed = f"{seed}_{time.time()}"
    hash_seed = hashlib.sha512(dynamic_seed.encode()).hexdigest()
    seed_matrix = np.array([int(char, 16) % 2 for char in hash_seed])
    seed_matrix = np.tile(seed_matrix, feature_length // len(seed_matrix) + 1)[:feature_length]
    return seed_matrix


def apply_cancelable_transform(features, seed_matrix):
    return features * seed_matrix

def create_cnn_model(input_shape, num_users, template_size):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.3),


        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.3),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.3),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(template_size, activation='sigmoid'),
        Dense(num_users, activation='softmax')

    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_cancelable_template(features, cnn_model):
    total_elements = features.size
    target_size = tz * tz

    if total_elements < target_size:
        padding = target_size - total_elements
        features_padded = np.pad(features, (0, padding), mode='constant', constant_values=0)
        features_reshaped = features_padded.reshape((1, tz, tz, 1))
        reshaped = True
    elif total_elements == target_size:
        features_reshaped = features.reshape((1, tz, tz, 1))
        reshaped = True
    else:
        features_resized = features[:target_size].reshape((1, tz, tz, 1))

        features_reshaped = features_resized
        reshaped = True

    if not reshaped:
        raise ValueError("Features cannot be reshaped or padded into a valid CNN input.")

    #print(f"Features reshaped to: {features_reshaped.shape}")
    templates = cnn_model.predict(features_reshaped)
    return templates


def enroll_users():
    number_of_user = 100
    fused_features = fuse_features(iris_features, fingerprint_features, face_features)
    user_seed = "secure_seed"
    enrolled_templates = []
    enrolled_labels = []


    reshaped_features = []

    target_size = tz * tz
    for features in fused_features:
        total_elements = features.size
        if total_elements < target_size:
            padding = target_size - total_elements
            features_padded = np.pad(features, (0, padding), mode='constant', constant_values=0)
            reshaped_features.append(features_padded.reshape(tz, tz, 1))
        elif total_elements == target_size:
            reshaped_features.append(features.reshape(tz, tz, 1))
        else:
            reshaped_features.append(features[:target_size].reshape(tz, tz, 1))

    reshaped_features = np.array(reshaped_features)
    #print(f"Reshaped features shape: {reshaped_features.shape}")

    labels = np.array([np.eye(number_of_user)[i] for i in range(len(reshaped_features))])
    #print(f"Labels shape: {labels.shape}")
    cnn_model = create_cnn_model((tz, tz, 1), num_users=number_of_user, template_size=template_size)
    cnn_model.fit(reshaped_features, labels, epochs=100, batch_size=16, verbose=1)
    cnn_model.save("cnn_model.h5")
    #print(f"Model saved as cnn_model.h5.")

    for i, user_label in enumerate(iris_labels):
        seed_matrix = generate_user_seed_matrix(user_seed, fused_features.shape[1])
        cancelable_features = apply_cancelable_transform(fused_features[i], seed_matrix)
        cancelable_template = generate_cancelable_template(cancelable_features, cnn_model)
        user_number=int(user_label.split('_')[1])
        _temp=np.concatenate((userKey.tolist()[user_number], cancelable_template[0]))
        #_temp = cancelable_template[0]
        #enrolled_templates.append(userKey.tolist()[user_number])
        enrolled_templates.append(_temp)
        enrolled_labels.append(user_label)

    np.savez("enrolled_templates.npz", templates=np.array(enrolled_templates), labels=np.array(enrolled_labels))
    print("Enrollment complete.")


def map_test_to_enrolled_label(test_label):
    if test_label.startswith("test_"):
        return test_label.replace("test_", "user_")
    return test_label


def get_test_raw_data():
    test_user_raw_data = {}
    if not os.path.exists('test_user_raw_data.npy'):
        test_query = {}

        test_folders = [os.path.join(test_root_folder, folder) for folder in os.listdir(test_root_folder) if
                        os.path.isdir(os.path.join(test_root_folder, folder))]

        test_user_raw_data = {}
        for test_folder in test_folders:
            userspecific = {}
            userName = os.path.basename(test_folder)
            iris_images = [f for f in os.listdir(test_folder) if f.endswith('.bmp')]
            fingerprint_images = [f for f in os.listdir(test_folder) if f.endswith('.tif')]
            face_images = [f for f in os.listdir(test_folder) if f.endswith('.jpg')]

            combined_iris_features = []
            combined_fingerprint_features = []
            combined_face_features = []
            #print("Extracting iris features from test images")
            for iris_image in iris_images:
                iris_image_path = os.path.join(test_folder, iris_image)
                iris_features = extract_iris_features(iris_image_path)
                combined_iris_features.append(iris_features)

            #print("Extracting fingerprint features from test images")
            for fingerprint_image in fingerprint_images:
                fingerprint_image_path = os.path.join(test_folder, fingerprint_image)
                base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                model = Model(inputs=base_model.input, outputs=base_model.output)
                fingerprint_features = extract_features_with_resnet50(model, fingerprint_image_path)
                combined_fingerprint_features.append(fingerprint_features)

            #print("Extracting face features from test images")
            for face_image in face_images:
                face_image_path = os.path.join(test_folder, face_image)
                face_features = extract_features_with_resnet50(model, face_image_path)
                combined_face_features.append(face_features)

            if len(combined_iris_features) == 0 or len(combined_fingerprint_features) == 0 or len(
                    combined_face_features) == 0:
                print("Skipping folder: ", test_folder)
                continue

            combined_iris_features = np.array(combined_iris_features)
            combined_fingerprint_features = np.array(combined_fingerprint_features)
            combined_face_features = np.array(combined_face_features)
            userspecific['iris'] = combined_iris_features
            userspecific['fp'] = combined_fingerprint_features
            userspecific['face'] = combined_face_features
            test_user_raw_data[userName] = userspecific
        np.save('test_user_raw_data.npy',test_user_raw_data)
    else:
        test_user_raw_data = np.load('test_user_raw_data.npy',allow_pickle=True).tolist()

    return test_user_raw_data



def generate_test_query():
    test_query={}
    test_raw_data=get_test_raw_data()
    cnn_model = tf.keras.models.load_model("cnn_model.h5")

    test_folders = [os.path.join(test_root_folder, folder) for folder in os.listdir(test_root_folder) if
                    os.path.isdir(os.path.join(test_root_folder, folder))]

    for test_folder in test_folders:
        try:
            userName = os.path.basename(test_folder)
            user_number = int(userName.split('_')[1])
            combined_iris_features=test_raw_data[userName]['iris']
            combined_fingerprint_features=test_raw_data[userName]['fp']
            combined_face_features=test_raw_data[userName]['face']

            fused_test_features = fuse_features(
                # combined_iris_features, combined_fingerprint_features[0], combined_face_features[0]
                np.mean(combined_iris_features, axis=0, keepdims=True),
                np.mean(combined_fingerprint_features, axis=0, keepdims=True),
                np.mean(combined_face_features, axis=0, keepdims=True)
            )

            print("Generating cancelable query for test features")
            cancelable_test_template = generate_cancelable_template(fused_test_features[0], cnn_model)
            _temp = np.concatenate((userKey.tolist()[user_number], cancelable_test_template[0]))
            #_temp = cancelable_test_template[0]
            #_temp = userKey.tolist()[user_number]

            test_query[userName]=_temp
        except:
            print(f'Some error occured for {userName}')
    np.save('test_query',test_query)

def Same():
    same_data=[]
    print("Authentication for Same Started...")
    enrolled_data = np.load("enrolled_templates.npz")
    enrolled_templates = enrolled_data['templates']
    enrolled_labels = enrolled_data['labels']

    test_query=np.load('test_query.npy',allow_pickle=True)

    #test_root_folder="D:\\Amber\\Multimodel Root\\CancelableTemplate_girls\\output_dataset\\test"

    test_folders = [os.path.join(test_root_folder, folder) for folder in os.listdir(test_root_folder) if
                    os.path.isdir(os.path.join(test_root_folder, folder))]

    for test_folder in test_folders:
        try:
            userName = os.path.basename(test_folder)
            index = np.where(enrolled_labels == userName.replace('test','user'))[0][0]
            saved_template=enrolled_templates[index]
            query=test_query.tolist()[userName]
            print(f'{userName} vs {enrolled_labels[index]} Matching start...')

            a, b = utility.pad_arrays(saved_template, query)

            # similarity = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
            dice_sim = utility.dice_coefficient(a, b)

            print(f'{userName} vs {enrolled_labels[index]} Matching complete. Scrore is {dice_sim}')
            same_data.append([enrolled_labels[index], userName, dice_sim * 100, 'Matched', 'Macthed' if dice_sim > threshold else 'Not Matched'])
        except:
            print(f'Error occured for {userName}')
    try:
        df = pd.DataFrame(same_data, columns=header)
        df.to_csv(f'same_result_keySize_{utility.string_size}.csv')
    except:
        print('Error occured in same result saving')

def Diff():
    print("Authentication for Diff Started...")
    enrolled_data = np.load("enrolled_templates.npz")
    enrolled_templates = enrolled_data['templates']
    enrolled_labels = enrolled_data['labels']
    test_query = np.load('test_query.npy', allow_pickle=True)

    test_folders = [os.path.join(test_root_folder, folder) for folder in os.listdir(test_root_folder) if
                    os.path.isdir(os.path.join(test_root_folder, folder))]
    diff_data=[]

    for test_folder in test_folders:
        try:
            userName = os.path.basename(test_folder)
            for label,template in zip(enrolled_labels,enrolled_templates):
                if label.replace('user','test')!=userName:
                    query = test_query.tolist()[userName]
                    print(f'{userName} vs {label} Matching start...')
                    a, b = utility.pad_arrays(template, query)
                    # similarity = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
                    dice_sim = utility.dice_coefficient(a, b)
                    print(f'{userName} vs {label} Matching complete. Scrore is {dice_sim}')
                    diff_data.append([label,userName,dice_sim*100,'Not Matched','Macthed' if dice_sim>threshold else 'Not Matched'])
        except:
            print(f'Error occured for {userName}')
    try:
        df=pd.DataFrame(diff_data,columns=header)
        df.to_csv(f'diff_result_keySize_{utility.string_size}.csv')
    except:
        print('Error in saving result file line 430')


print('Enrollment Start')
start_time=time.time()
enroll_users()
end_time=time.time()
print(f'Total time to enroll User : {end_time-start_time}')
print('Enrollment Complete. Preparing Test Query')
start_time=time.time()
generate_test_query()
print('Matching Start Same')
end_time=time.time()
print(f'Total time to Create Query: {end_time-start_time}')
start_time=time.time()
Same()
end_time=time.time()
print(f'Total time to Genarete Same result: {end_time-start_time}')

print('Matching Start Diff')
start_time=time.time()
Diff()
end_time=time.time()
print(f'Total time to Genarete Diff result: {end_time-start_time}')
