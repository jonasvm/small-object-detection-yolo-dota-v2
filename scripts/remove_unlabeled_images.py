import os
import shutil

images_dir = "/home/jonasvm/docker-images/dota_dataset/images/train"
labels_dir = "/home/jonasvm/docker-images/dota_dataset/labelTxt/train"
unlabeled_images_dir = "/home/jonasvm/docker-images/dota_dataset/unlabeled_images"

os.makedirs(unlabeled_images_dir, exist_ok=True)

valid_image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def has_labels(label_path):
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            content_lines = [line.strip() for line in lines[2:] if line.strip()]
            return len(content_lines) > 0
    except:
        return False

for img in os.listdir(images_dir):
    if os.path.splitext(img)[1].lower() in valid_image_exts:
        base_name = os.path.splitext(img)[0]
        label_file = os.path.join(labels_dir, base_name + ".txt")
        if os.path.isfile(label_file) and not has_labels(label_file):
            src_path = os.path.join(images_dir, img)
            dst_path = os.path.join(unlabeled_images_dir, img)
            shutil.move(src_path, dst_path)
            print(f"Movido {img} para unlabeled_images")

print("Finalizado!")
