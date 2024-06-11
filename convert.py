import cv2
import os

def convert_bgr_to_rgb(input_video_path, output_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open input video file: {input_video_path}")
        return

    # Get the width, height, and frame rate of the input video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use 'MJPG' codec for JPEG
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Write the frame to the output video
        out.write(rgb_frame)

        frame_number += 1
        print(f"Processed frame {frame_number}/{frame_count}")

    # Release the video capture and writer objects
    cap.release()
    out.release()

    print(f"Conversion complete. Output video saved to: {output_video_path}")

def convert_all_videos_in_folder(input_folder, output_folder):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder does not exist: {input_folder}")
        return

    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all video files in the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print("No video files found in the input folder.")
        return

    # Convert each video
    for video_file in video_files:
        input_video_path = os.path.join(input_folder, video_file)
        output_video_path = os.path.join(output_folder, f"rgb_{video_file}")
        print(f"Converting video: {input_video_path} -> {output_video_path}")
        convert_bgr_to_rgb(input_video_path, output_video_path)

if __name__ == "__main__":
    input_folder = "/Users/cam/Programming/StereoMatchingTool/stereo_imgs/LabTest/Right"  # Replace with your input folder path
    output_folder = "/Users/cam/Programming/StereoMatchingTool/stereo_imgs/LabTest/Right"  # Replace with your desired output folder path
    convert_all_videos_in_folder(input_folder, output_folder)
