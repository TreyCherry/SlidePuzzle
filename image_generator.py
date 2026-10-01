import torch
from diffusers import DiffusionPipeline
from PIL import Image
import threading
import queue
import os
import json
from datetime import datetime

class ImageGenerator:
    def __init__(self, save_directory):
        self.pipe = DiffusionPipeline.from_pretrained("stabilityai/sd-turbo")
        self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self.queue = queue.Queue(maxsize=2)
        self.thread = None
        self.current_themes = []
        self.save_directory = save_directory
        self.ensure_save_directory()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_counter = self.get_last_image_id() + 1

    def ensure_save_directory(self):
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)

    def get_last_image_id(self):
        image_files = [f for f in os.listdir(self.save_directory) if f.endswith('.png')]
        if not image_files:
            return 0
        return max(int(f.split('_')[0]) for f in image_files)

    def generate_image(self, prompt):
        with torch.autocast("cuda"):
            print("prompt: ", prompt)
            image = self.pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        return image

    def start_generation(self, prompt, themes):
        self.current_themes = themes
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._generate_and_queue, args=(prompt,))
            self.thread.start()

    def _generate_and_queue(self, prompt):
        #while True:
            #if self.queue.qsize() < 2:
        image = self.generate_image(prompt)
        self.save_image(image, prompt)
        self.queue.put((self.image_counter, image))
            #else:
            #    break

    def save_image(self, image, prompt):
        image_filename = f"{self.image_counter:06d}_{self.session_id}.png"
        image_path = os.path.join(self.save_directory, image_filename)
        image.save(image_path)

        metadata = {
            "id": self.image_counter,
            "session_id": self.session_id,
            "prompt": prompt,
            "themes": self.current_themes,
            "timestamp": datetime.now().isoformat()
        }
        metadata_filename = f"{self.image_counter:06d}_{self.session_id}.json"
        metadata_path = os.path.join(self.save_directory, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        self.image_counter += 1

    def get_generated_image(self):
        if not self.queue.empty():
            return self.queue.get()
        return None

    def is_generating(self):
        return self.thread is not None and self.thread.is_alive()

    def get_image_history(self):
        history = []
        for filename in sorted(os.listdir(self.save_directory)):
            if filename.endswith('.json'):
                with open(os.path.join(self.save_directory, filename), 'r') as f:
                    metadata = json.load(f)
                image_filename = filename.replace('.json', '.png')
                image_path = os.path.join(self.save_directory, image_filename)
                if os.path.exists(image_path):
                    history.append((metadata['id'], metadata['prompt'], Image.open(image_path), metadata))
        return history


#Version 2
# import torch
# from diffusers import DiffusionPipeline
# from PIL import Image
# import threading
# import queue

# class ImageGenerator:
    # def __init__(self):
        # self.pipe = DiffusionPipeline.from_pretrained("stabilityai/sd-turbo")
        # self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        # self.queue = queue.Queue(maxsize=2)  # Keep at most 2 images in the queue
        # self.thread = None
        # self.current_themes = []
        # self.image_history = []
        # self.image_counter = 0

    # def generate_image(self, prompt):
        # with torch.autocast("cuda"):
            # print("prompt: ", prompt)
            # image = self.pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        # return image

    # def start_generation(self, prompt, themes):
        # self.current_themes = themes
        # if self.thread is None or not self.thread.is_alive():
            # self.thread = threading.Thread(target=self._generate_and_queue, args=(prompt,))
            # self.thread.start()

    # def _generate_and_queue(self, prompt):
        # while True:
            # if self.queue.qsize() < 2:
                # image = self.generate_image(prompt)
                # self.image_counter += 1
                # self.image_history.append((self.image_counter, prompt, image))
                # self.queue.put((self.image_counter, image))
            # else:
                # break

    # def get_generated_image(self):
        # if not self.queue.empty():
            # return self.queue.get()
        # return None

    # def is_generating(self):
        # return self.thread is not None and self.thread.is_alive()

    # def get_image_history(self):
        # return self.image_history

# Version 1
# import torch
# from diffusers import DiffusionPipeline
# from PIL import Image
# import threading
# import queue

# class ImageGenerator:
    # def __init__(self):
    # #def __init__(self, model_path):
        # #self.model_path = model_path
        # self.pipe = DiffusionPipeline.from_pretrained("stabilityai/sd-turbo")
        # self.pipe = self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        # self.queue = queue.Queue()
        # self.thread = None

    # def generate_image(self, prompt):
        # with torch.autocast("cuda"):
            # print("prompt: ", prompt)
            # image = self.pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        # return image

    # def start_generation(self, prompt):
        # if self.thread is None or not self.thread.is_alive():
            # self.thread = threading.Thread(target=self._generate_and_queue, args=(prompt,))
            # self.thread.start()

    # def _generate_and_queue(self, prompt):
        # image = self.generate_image(prompt)
        # self.queue.put(image)

    # def get_generated_image(self):
        # if not self.queue.empty():
            # return self.queue.get()
        # return None

    # def is_generating(self):
        # return self.thread is not None and self.thread.is_alive()