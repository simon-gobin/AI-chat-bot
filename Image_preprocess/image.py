import os
import cv2
import random
from PIL import Image as PILImage

class ImageExtractor:
    def __init__(self, image_origin_folder, image_output_folder, video_images=100):
        self.image_origin_folder = image_origin_folder
        self.image_output_folder = image_output_folder
        self.video_images = video_images

        os.makedirs(self.image_output_folder, exist_ok=True)

    def video_extract(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < self.video_images:
            raise ValueError("Number of frames requested exceeds total frames in video.")

        selected_frames = sorted(random.sample(range(total_frames), self.video_images))
        extracted = 0

        for frame_no in selected_frames:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            success, image = vidcap.read()
            if success:
                resized = cv2.resize(image, (768, 768))
                filename = f"frame_{extracted:03d}.jpg"
                output_path = os.path.join(self.image_output_folder, filename)
                cv2.imwrite(output_path, resized)
                extracted += 1

                if extracted % 5 == 0:
                    print(f"[INFO] {extracted} frames extracted to {self.image_output_folder}")

        vidcap.release()
        print(f"[DONE] {extracted} frames extracted to {self.image_output_folder}")

    def image_convert(self, img_path):
        try:
            img = PILImage.open(img_path)
            img = img.resize((768, 768))
            filename = os.path.basename(img_path)
            output_path = os.path.join(self.image_output_folder, filename)
            img.save(output_path)
        except Exception as e:
            print(f"[ERROR] Cannot convert {img_path}: {e}")

    def run(self):
        for file in os.listdir(self.image_origin_folder):
            full_path = os.path.join(self.image_origin_folder, file)
            if file.endswith('.mp4'):
                print(f"[VIDEO] Processing {file}")
                self.video_extract(full_path)
            elif file.endswith('.jpg'):
                print(f"[IMAGE] Processing {file}")
                self.image_convert(full_path)
            else:
                print(f"[SKIP] Skipping unsupported file: {file}")
