import cv2
import os 
import numpy as np
from pathlib import Path
import tensorflow as tf
import random 
import copy
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageProcessing:
    # display one images of each class
    def display_images(self, images, labels):
        m = []
        for i, label in enumerate(labels):
            if label not in m:
                m.append(label)
                cv2.imshow(f"Class: {label}", images[i])
                cv2.waitKey(0) 
                cv2.destroyAllWindows()

    def grayscale_conversion(self, images):
        grayscale_images  = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
        return grayscale_images

    def resize(self, images, height, width):
        resized_images = [cv2.resize(image, (height, width)) for image in images]
        return resized_images

    def blur(self, images):
        blurred_images = [cv2.GaussianBlur(image, (5, 5), 0) for image in images]
        return blurred_images

    def gaussian_noise_add(self, images):
        mean = 0
        stddev = 5
        noisy_images = [image + np.random.normal(mean, stddev, image.shape).astype(np.uint8) for image in images]
        return noisy_images

    def denoise(self, images):
        denoised_images = [cv2.fastNlMeansDenoising(image, None, h=10, templateWindowSize=7
                                           , searchWindowSize=21) for image in images]
        return denoised_images

    def random_rotation(self, image, max_angle=15):
        angle = np.random.uniform(-max_angle, max_angle) 
        rows, cols = image.shape[:2] 
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1) 
        rotated_image = cv2.warpAffine(image, M, (cols, rows)) 
        return rotated_image

    def random_flip(self, image): 
        flip_code = np.random.choice([-1, 0, 1])
        return cv2.flip(image, flip_code) 

    def random_crop(self, image, crop_size=(150, 150)): 
        rows, cols = image.shape[:2] 
        x = np.random.randint(0, cols - crop_size[0] + 1) 
        y = np.random.randint(0, rows - crop_size[1] + 1) 
        cropped_image = image[y:y+crop_size[1], x:x+crop_size[0]] 
        return cropped_image 

    def augmentation(self, images, labels): 
        augmented_images = copy.deepcopy(images)
        augmented_labels = copy.deepcopy(labels)

        for image, label in zip(images, labels): 
            augmented_limit = random.choice([2, 3, 4]) 
            for _ in range(augmented_limit): 
                func = random.choice([self.random_rotation, self.random_flip, self.random_crop]) 
                augmented_image = func(image) 
                augmented_images.append(augmented_image) 
                augmented_labels.append(label) 
        return augmented_images, augmented_labels

    def edge_detection(self, images):
        edges = [cv2.Canny(image, threshold1=100, threshold2=200) for image in images]
        return edges

    def hist_equalization(self, images):
        hist_eq_images = [cv2.equalizeHist(image) for image in images]
        return hist_eq_images

    def global_threshold(self, images):
        global_thresholded_images = []
        for image in images:
            _, global_thresholded = cv2.threshold(image, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
            global_thresholded_images.append(global_thresholded)
        return global_thresholded_images

    def adaptive_threshold(self, images):
        adaptive_threholded_images = []
        for image in images:
            adaptive_thresholded = cv2.adaptiveThreshold(image, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                                    thresholdType=cv2.THRESH_BINARY, blockSize=11, C=2)
            adaptive_threholded_images.append(adaptive_thresholded)
        return adaptive_threholded_images

    def save_processed_files(self, images, labels):
        output_dir = 'cv_output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        count = 1
        for image, label in zip(images, labels):
            path = os.path.join(output_dir, str(label))
            
            if not os.path.exists(path):
                os.makedirs(path)
                count = 1
            
            image_file_path = os.path.join(path, f'processed_image_{count}.jpg')

            cv2.imwrite(image_file_path, image)
            count+=1


if __name__=='__main__':
    root_dir = Path('Sample_Pokemon/')
    images = []
    labels = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', 'bmp')):
                image_path = Path(root) / file
                class_label = image_path.parts[-2]
                image = cv2.imread(str(image_path))
                try:
                    images.append(image)
                    labels.append(class_label)
                except:
                    logging.info(f"Failed to load image: {image_path}")

    img_processor = ImageProcessing()

    # 1. conversion to grayscale
    grayscale_images = img_processor.grayscale_conversion(images)
    
    # 2. Resize images
    resized_images = img_processor.resize(grayscale_images, height=256, width=256)

    # 3. Blur images
    blurred_images = img_processor.blur(resized_images)

    # 4. Noise Adding 
    added_noise_images = img_processor.gaussian_noise_add(blurred_images)

    # 5. Denoising 
    denoised_images = img_processor.denoise(added_noise_images)

    logging.info("Denoised Images Length:", len(denoised_images))

    # 6. Augmentation (Rotation, Flip, Crop)
    augmented_images, augmented_labels = img_processor.augmentation(denoised_images, labels)

    logging.info("Augmented Images Length:", len(augmented_images))

    # 7. Detect Edges
    edge_detected_images = img_processor.edge_detection(augmented_images)
    
    # 8. histogram equilization
    hist_eq_images = img_processor.hist_equalization(edge_detected_images)

    # 9. global  thresholding 
    global_thresholded_images = img_processor.global_threshold(hist_eq_images)

    # 10. adaptive thresholding
    adaptive_thresholded_images = img_processor.adaptive_threshold(global_thresholded_images)

    # 11. display the processed images 
    img_processor.display_images(adaptive_thresholded_images, augmented_labels)

    # 12. Saves the processed images
    img_processor.save_processed_files(adaptive_thresholded_images, augmented_labels)




