# Ammeter Image Reader

This tool automatically reads ammeter values from images by detecting the needle position and calculating the reading based on calibration images.

## Features

- Detects ammeter needle position using computer vision techniques
- Calculates reading values based on standard reference images
- Outputs annotated images with reading values

## Requirements

- Python 3.6 or higher
- OpenCV
- NumPy

## Installation

1. Clone or download this repository to your local machine
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Preparing Your Data

1. Place your standard reference images in the `Standard/` folder:
   - `0.png` - Reference image showing the needle at 0 position
   - `50.png` - Reference image showing the needle at 50 position

2. Place your ammeter images to be processed in the `Images/` folder

## Running the Program

### Option 1: Using the run script
Simply execute the run.sh script:
```
./run.sh
```

### Option 2: Running manually
```
python read_ammeter_img.py --input_img_path Images/ --output_img_path Result/
```

## Output

The processed images will be saved in the `Results/` folder. Each output image contains:
- A red line showing the detected needle position
- A text box displaying the calculated reading value

## Important Notes

- Camera angle and distance from the ammeter must remain fixed when taking all images
- For accurate readings, ensure the camera position is consistent between calibration images and measurement images
- Avoid shadows or reflections on the ammeter face that might interfere with needle detection
- Ensure good lighting conditions for best results
