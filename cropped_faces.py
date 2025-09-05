import os
import pandas as pd
import face_recognition
import cv2

# csv_path = './Dataset/label_val.csv'
csv_path = './Dataset/label_val_cropped_err.csv'
img_folder = './Dataset/val'
save_folder = './Dataset/val_cropped'
error_log_path = 'error_log.txt'

os.makedirs(save_folder, exist_ok=True)

df = pd.read_csv(csv_path, sep=',', encoding='gbk')

error_files = []

for idx, row in df.iterrows():
    filename = row['file']
    img_path = os.path.join(img_folder, filename)
    
    if not os.path.exists(img_path):
        print(f"Cannot find: {img_path}")
        error_files.append(filename)
        continue

    try:
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)

        if not face_locations:
            print(f"no faces in: {filename}")
            error_files.append(filename)
            continue

        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_image = image[top:bottom, left:right]
            face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(save_folder, f"{os.path.splitext(filename)[0]}.jpg")
            cv2.imwrite(save_path, face_image_bgr)
            print(f"Save: {save_path}")

    except Exception as e:
        print(f"Fail: {filename} Err: {e}")
        error_files.append(filename)

with open(error_log_path, 'w', encoding='utf-8') as f:
    for name in error_files:
        f.write(name + '\n')

print(f"Done, there is {len(error_files)} fails, recorded in {error_log_path}")
