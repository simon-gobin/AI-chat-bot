import os
import cv2
import json
import random
from PIL import Image as PILImage
from transformers import BlipProcessor, BlipForConditionalGeneration

class ImageExtractor:
    def __init__(self, image_origin_folder, image_output_folder, video_images=100, image_size=768):
        self.image_origin_folder = image_origin_folder
        self.image_output_folder = image_output_folder
        self.video_images = video_images
        self.image_size = image_size

        os.makedirs(self.image_output_folder, exist_ok=True)

    def load_model(self):
        print("[INFO] Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

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
                resized = cv2.resize(image, (self.image_size , self.image_size ))
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
            img = img.resize((self.image_size , self.image_size ))
            filename = os.path.basename(img_path)
            output_path = os.path.join(self.image_output_folder, filename)
            img.save(output_path)
        except Exception as e:
            print(f"[ERROR] Cannot convert {img_path}: {e}")

    def generate_captions(self):
        print("[INFO] Generating captions for all images...")
        captions = []
        for file in os.listdir(self.image_output_folder):
            if file.endswith('.jpg'):
                img_path = os.path.join(self.image_output_folder, file)
                image = PILImage.open(img_path).convert("RGB")
                text = "a photography of"
                inputs = self.processor(image, text, return_tensors="pt").to("cuda")
                out = self.model.generate(**inputs)
                text = self.processor.decode(out[0], skip_special_tokens=True)
                captions.append({"file_name": file, "text": text})

        with open(os.path.join(self.image_output_folder, "captions.jsonl"), 'w') as f:
            for entry in captions:
                f.write(json.dumps(entry) + "\n")

        print(f"[DONE] Captions saved to {self.image_output_folder}/captions.jsonl")

    def run(self):
        self.load_model()

        for root, _, files in os.walk(self.image_origin_folder):
            for file in files:
                full_path = os.path.join(root, file)
                if file.endswith('.mp4'):
                    print(f"[VIDEO] Processing {full_path}")
                    self.video_extract(full_path)
                elif file.endswith(('.jpg', '.jpeg', '.png')):
                    print(f"[IMAGE] Processing {full_path}")
                    self.image_convert(full_path)
                else:
                    print(f"[SKIP] Skipping unsupported file: {file}")


        self.generate_captions()

if __name__ == "__main__":
    class_ = ImageExtractor('/home/simon/Downloads', '/home/simon/image_process' )
    class_.run()
