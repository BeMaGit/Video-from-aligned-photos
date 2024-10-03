import cv2
import os
import numpy as np
from PIL import Image, ExifTags
from datetime import datetime
from moviepy.editor import ImageSequenceClip

def get_image_timestamp(image_path):
    """Extract the timestamp from the EXIF metadata of the image."""
    img = Image.open(image_path)
    exif_data = img._getexif()
    for tag, value in ExifTags.TAGS.items():
        if value == 'DateTimeOriginal':
            date_taken = exif_data.get(tag)
            if date_taken:
                return datetime.strptime(date_taken, "%Y:%m:%d %H:%M:%S")
    return None

def align_images(image1, image2):
    """Align image2 to image1 using feature-based matching (SIFT)."""
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of matched points
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography matrix to warp the second image
    matrix, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    height, width = image1.shape[:2]
    aligned_image = cv2.warpPerspective(image2, matrix, (width, height))
    
    return aligned_image

def create_video(image_paths, timestamps, total_duration=120, output_path="output_video.mp4"):
    """Generate a video from the aligned images based on their timestamps."""
    # Calculate time differences between images
    time_deltas = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
    total_time_span = sum(time_deltas)
    
    # Calculate frame durations proportional to the real time span
    frame_durations = [(delta / total_time_span) * total_duration for delta in time_deltas]

    # Align images and collect them into a list
    aligned_images = []
    first_image = cv2.imread(image_paths[0])
    aligned_images.append(first_image)  # No need to align the first image

    for i in range(1, len(image_paths)):
        img1 = cv2.imread(image_paths[i-1])
        img2 = cv2.imread(image_paths[i])
        aligned_img = align_images(img1, img2)
        aligned_images.append(aligned_img)

    # Save aligned images to a temporary folder
    temp_dir = "temp_frames"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    temp_image_paths = []
    for i, img in enumerate(aligned_images):
        temp_img_path = os.path.join(temp_dir, f"frame_{i}.jpg")
        cv2.imwrite(temp_img_path, img)
        temp_image_paths.append(temp_img_path)
    
    # Create video with moviepy, setting the correct frame durations
    clip = ImageSequenceClip(temp_image_paths, durations=frame_durations + [frame_durations[-1]])
    clip.set_duration(total_duration)
    clip.write_videofile(output_path, fps=24)

    # Cleanup temporary images
    for temp_img_path in temp_image_paths:
        os.remove(temp_img_path)
    os.rmdir(temp_dir)

def main(image_folder):
    # Get all image file paths from the folder
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])

    # Extract timestamps from images
    timestamps = [get_image_timestamp(path) for path in image_paths]

    # Check if all timestamps are valid
    if None in timestamps:
        print("Some images are missing EXIF timestamps.")
        return

    # Create video from images
    create_video(image_paths, timestamps)

if __name__ == "__main__":
    image_folder = "path/to/your/images"  # Replace with the path to your image folder
    main(image_folder)
