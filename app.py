import io
import base64
import io

import cv2
import numpy as np
import pydicom
from PIL import Image
from dash import Dash, html, dcc, Output, Input, State, no_update
from ultralytics import YOLO

app = Dash(__name__)
app.title = "DICOM to PNG Viewer"

app.layout = html.Div(
    style={"backgroundColor": "#f9f9f9", "padding": "2rem", "fontFamily": "Arial, sans-serif"},
    children=[
        html.H1("DICOM to PNG Viewer", style={"textAlign": "center", "color": "#333"}),

        dcc.Upload(
            id='upload-dicom-folder',
            children=html.Div([
                "Drag and Drop or ",
                html.A("Select DICOM Files")
            ]),
            style={
                "width": "100%",
                "height": "150px",
                "lineHeight": "150px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "10px",
                "textAlign": "center",
                "backgroundColor": "#ffffff",
                "color": "#333",
                "marginBottom": "20px",
            },
            multiple=True
        ),

        html.Div(
            id='output-images',
            style={
                "whiteSpace": "nowrap",
                "overflowX": "auto",
                "padding": "1rem",
                "backgroundColor": "#ffffff",
                "borderRadius": "10px",
                "boxShadow": "0 4px 8px rgba(0,0,0,0.05)"
            }
        )
    ]
)

model = YOLO("yolo11n-tumor-luca.pt")


def _draw_bounding_boxes(img, x1, y1, x2, y2):
    """
    Draw bounding boxes on image and add detection to all_detections
    """

    # Draw rectangle - thicker bright green line
    GREEN = (0, 255, 0)  # BGR format - bright green
    cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 3)

    # Add ID with smaller text and better positioning
    label = f"Abnormality"

    # Calculate text size to create a background - reduced font size to 1.0 and thickness to 2
    font_size = 1.0
    thickness = 2
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]

    # Draw a filled rectangle as background for text
    cv2.rectangle(img,
                  (x1, y1 - text_size[1] - 5),
                  (x1 + text_size[0], y1),
                  (0, 0, 0),
                  -1)  # -1 means filled

    # Draw text with bright green - using smaller font size and thickness
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, GREEN, thickness)


def dicom_to_png_bytes(dicom_bytes):
    try:
        dcm = pydicom.dcmread(io.BytesIO(dicom_bytes))
        pixel_array = dcm.pixel_array

        # Normalize pixel values
        pixel_array = pixel_array.astype(float)
        pixel_array -= np.min(pixel_array)
        pixel_array /= np.max(pixel_array)
        pixel_array *= 255.0

        image = Image.fromarray(pixel_array.astype(np.uint8)).convert("RGB")
        pixel_array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        results = model(image)

        for result in results:
            boxes = result.boxes.xyxy  # [x1, y1, x2, y2] format
            confs = result.boxes.conf  # Confidence scores
            classes = result.boxes.cls  # Class IDs
            print(f"Detected {len(boxes)} objects")

            # Draw bounding boxes with confidence > threshold
            for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                if conf > 0.5:
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)

                    # Calculate physical dimensions if pixel spacing is available
                    physical_size = None
                    if dcm["PixelSpacing"] is not None:
                        pixel_spacing = dcm["PixelSpacing"]
                        width_mm = (x2 - x1) * float(pixel_spacing[0])
                        height_mm = (y2 - y1) * float(pixel_spacing[1])
                        physical_size = [round(width_mm, 2), round(height_mm, 2)]

                    # Store detection data
                    detection_data = {
                        "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                        "physical_size_mm": physical_size,
                        "conf": float(conf),
                        "cls": int(cls),
                        # "metadata": all_metadata[f]
                    }

                    _draw_bounding_boxes(pixel_array, x1, y1, x2, y2)

        image = Image.fromarray(pixel_array.astype(np.uint8)).convert("RGB")

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        encoded_png = base64.b64encode(buffered.getvalue()).decode()
        return encoded_png
    except Exception as e:
        print(f"Error converting DICOM: {e}")
        return None


@app.callback(
    Output('output-images', 'children'),
    Input('upload-dicom-folder', 'contents'),
    State('upload-dicom-folder', 'filename')
)
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        images = []
        for content, name in zip(list_of_contents, list_of_names):
            content_type, content_string = content.split(',')
            dicom_bytes = base64.b64decode(content_string)
            png_encoded = dicom_to_png_bytes(dicom_bytes)
            if png_encoded:
                img_element = html.Img(
                    src=f"data:image/png;base64,{png_encoded}",
                    style={
                        "height": "200px",  # Fixed height
                        "marginRight": "1rem",
                        "borderRadius": "10px",
                        "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                        "display": "inline-block",
                    }
                )
                images.append(img_element)
        return images
    return no_update


if __name__ == '__main__':
    app.run(debug=True)
