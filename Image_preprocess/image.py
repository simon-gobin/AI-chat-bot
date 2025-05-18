import os
import cv2
import json
import random
from PIL import Image as PILImage
from transformers import BlipProcessor, BlipForConditionalGeneration
import sys
import argparse
import logging

# CLI or ENV config for debug mode
parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument("--input", default="/input", help="Input directory path")
parser.add_argument("--output", default="/output", help="Output directory path")
args, unknown = parser.parse_known_args()


DEBUG_ENV = os.getenv("DEBUG", "0") == "1"
DEBUG_MODE = args.debug or DEBUG_ENV

# Set up logging to both file and console
log_path = "/input/image_extractor.log"
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)

class ImageExtractor:
    def __init__(self, image_origin_folder, image_output_folder, video_images=100, image_size=1024):
        self.image_origin_folder = image_origin_folder
        self.image_output_folder = image_output_folder
        self.video_images = video_images
        self.image_size = image_size

        os.makedirs(self.image_output_folder, exist_ok=True)

    def load_model(self):
        log.info("[INFO] Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

    def check_out_put(self):
        if not os.path.exists(self.image_output_folder):
            log.info(f'no out put create one {self.image_output_folder}')
            os.makedirs(self.image_output_folder)
        else:
            log.info(f'out put found {self.image_output_folder}')


    def video_extract(self, video_path):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        vidcap = cv2.VideoCapture(video_path)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < self.video_images:
            log.warning(f"[WARNING] Video has only {total_frames} frames. Reducing to fit.")
            self.video_images = total_frames

        selected_frames = sorted(random.sample(range(total_frames), self.video_images))
        extracted = 0

        for frame_no in selected_frames:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            success, image = vidcap.read()
            if success:
                resized = cv2.resize(image, (self.image_size , self.image_size ))
                filename = f"frame_{video_name}_{extracted:03d}.jpg"
                output_path = os.path.join(self.image_output_folder, filename)
                cv2.imwrite(output_path, resized)
                extracted += 1

                if extracted % 5 == 0:
                    log.info(f"[INFO] {extracted} frames extracted to {self.image_output_folder}")

        vidcap.release()
        log.info(f"[DONE] {extracted} frames extracted to {self.image_output_folder}")

    def image_convert(self, img_path):
        try:
            img = PILImage.open(img_path)
            img = img.resize((self.image_size , self.image_size ))
            filename = os.path.basename(img_path)
            output_path = os.path.join(self.image_output_folder, filename)
            img.save(output_path)
        except Exception as e:
            log.error(f"[ERROR] Cannot convert {img_path}: {e}")

    def generate_captions(self):
        log.info("[INFO] Generating captions for all images...")
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

        log.info(f"[DONE] Captions saved to {self.image_output_folder}/captions.jsonl")

    def run(self):
        self.load_model()
        self.check_out_put()

        processed_any = False

        for root, _, files in os.walk(self.image_origin_folder):
            for file in files:
                full_path = os.path.join(root, file)
                log.debug(f'[DEBUG] Found: {full_path}')

                if os.path.islink(full_path):
                    log.error(f"[SKIP] Symlink: {file}")
                    continue

                if file.lower().endswith('.mp4'):
                    log.info(f"[VIDEO] Processing {full_path}")
                    self.video_extract(full_path)
                    processed_any = True
                elif file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    log.debug(f"[IMAGE] Processing {full_path}")
                    self.image_convert(full_path)
                    processed_any = True
                else:
                    log.error(f"[SKIP] Unsupported file: {file}")


        if not processed_any:
            log.warning(f"[WARNING] No valid images or videos found in {self.image_origin_folder}")

        self.generate_captions()

if __name__ == "__main__":
    class_ = ImageExtractor(args.input, args.output)
    class_.run()

