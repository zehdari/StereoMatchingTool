
# Stereo Vision Disparity Adjustment Tool

This tool provides a graphical interface for adjusting the disparity map parameters of a stereo vision system. It includes visualizations for depth maps and point clouds and allows for real-time tuning of stereo vision parameters.

## Features

- **Disparity Map Adjustment**: Adjust parameters like numDisparities, blockSize, preFilterCap, uniquenessRatio, etc.
- **Depth Map Visualization**: View the depth map in grayscale or color.
- **Point Cloud Visualization**: Visualize the 3D point cloud generated from the disparity map.
- **Image Display Modes**: Switch between RGB, BGR, and Greyscale display modes for rectified images.
- **Algorithm Toggle**: Switch between SGBM and BM stereo matching algorithms.
- **WLS Filter**: Option to apply WLS (Weighted Least Squares) filter to the disparity map.

## Configuration

Before running the tool, create a `config/config.yml` file with the following structure:

```yaml
left_folder: path/to/left/images
right_folder: path/to/right/images
left_camera_config: path/to/left/camera/config.yml
right_camera_config: path/to/right/camera/config.yml
baseline: 120 # mm
```

**Note**: The camera configurations should be in OpenCV format.

## Setting Up the Environment

To set up the virtual environment and install the required dependencies, follow these steps:

1. **Create a Virtual Environment**:

    ```bash
    python -m venv venv
    ```

2. **Activate the Virtual Environment**:
    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

3. **Install the Required Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Initialize and Run**:

    ```bash
    python stereo_vision_app.py
    ```

2. **Adjust Parameters**: Use the sliders on the left panel to adjust the disparity map parameters.
3. **Switch Modes**: Use the buttons on the top panel to switch display modes, toggle depth map color, change algorithms, and apply the WLS filter.

## Running the Application

To run the application, execute the script:

```bash
python stereo_matching.py
```

The application window will open, allowing you to adjust stereo vision parameters and visualize the results in real-time.
