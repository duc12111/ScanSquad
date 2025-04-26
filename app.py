import io
import base64
import io
from os import listdir
from os.path import join
import os
import time
import threading
import queue
from collections import OrderedDict
from datetime import datetime
import cv2
import numpy as np
import pydicom
from PIL import Image
from dash import Dash, html, dcc, Output, Input, State, no_update, callback_context, ALL
from dash.exceptions import PreventUpdate
from ultralytics import YOLO
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.callbacks.manager import get_openai_callback
import json
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

# Set the OpenAI API key
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Queue for background report generation
report_queue = queue.Queue()
report_results = {}

app = Dash(__name__)
app.title = "DICOM to PNG Viewer"

# Define animations using keyframes in a style dictionary
dot_animation_style = {
    "animation": "dot-animation 1.4s infinite",
    "opacity": "0.3"
}

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
                "boxShadow": "0 4px 8px rgba(0,0,0,0.05)",
                "marginTop": "20px"
            }
        ),

        # Simplified modal for full-size image viewing
        html.Div(
            id='image-modal',
            style={
                "display": "none",
                "position": "fixed",
                "zIndex": "1000",
                "left": "0",
                "top": "0",
                "width": "100%",
                "height": "100%",
                "overflow": "auto",
                "backgroundColor": "rgba(0,0,0,0.9)",
                "textAlign": "center"
            },
            children=[
                # Close button
                html.Button(
                    "×",
                    id="close-modal",
                    style={
                        "position": "absolute",
                        "top": "15px",
                        "right": "35px",
                        "fontSize": "40px",
                        "fontWeight": "bold",
                        "backgroundColor": "transparent",
                        "border": "none",
                        "color": "white",
                        "cursor": "pointer"
                    }
                ),

                # Next button
                html.Button(
                    "→",
                    id="next-image",
                    style={
                        "position": "absolute",
                        "top": "50%",
                        "right": "35px",
                        "fontSize": "40px",
                        "fontWeight": "bold",
                        "backgroundColor": "rgba(0,0,0,0.3)",
                        "border": "none",
                        "color": "white",
                        "cursor": "pointer",
                        "borderRadius": "50%",
                        "width": "60px",
                        "height": "60px",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "transform": "translateY(-50%)"
                    }
                ),

                # Previous button
                html.Button(
                    "←",
                    id="prev-image",
                    style={
                        "position": "absolute",
                        "top": "50%",
                        "left": "35px",
                        "fontSize": "40px",
                        "fontWeight": "bold",
                        "backgroundColor": "rgba(0,0,0,0.3)",
                        "border": "none",
                        "color": "white",
                        "cursor": "pointer",
                        "borderRadius": "50%",
                        "width": "60px",
                        "height": "60px",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "transform": "translateY(-50%)"
                    }
                ),

                # Simple image container
                html.Img(
                    id="modal-image",
                    style={
                        "maxWidth": "90%",
                        "maxHeight": "90%",
                        "margin": "40px auto",
                        "display": "block"
                    }
                ),

                # Store current image index
                dcc.Store(id="current-image-index", data=0),

                # Store total number of images
                dcc.Store(id="total-images", data=0)
            ]
        ),

        # Report output area with loading indicator
        html.Div([
            html.H2("MRI Analysis Report", style={"color": "#333", "marginBottom": "10px"}),
            dcc.Textarea(
                id="report-output",
                style={
                    "width": "100%",
                    "height": "400px",
                    "padding": "20px",
                    "backgroundColor": "white",
                    "borderRadius": "10px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.05)",
                    "whiteSpace": "pre-line",
                    "fontFamily": "Arial, sans-serif",
                    "fontSize": "14px",
                    "lineHeight": "1.5",
                    "border": "1px solid #4CAF50",
                    "cursor": "text"
                },
                value="Upload images to generate a report",
                readOnly=False
            ),
            html.Div([
                html.Button(
                    "Export PDF",
                    id="export-pdf",
                    style={
                        "marginTop": "15px",
                        "marginRight": "10px",
                        "padding": "10px 20px",
                        "backgroundColor": "#2196F3",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                        "fontSize": "14px",
                        "fontWeight": "bold"
                    }
                ),
                html.Button(
                    "Save",
                    id="toggle-edit-mode",
                    style={
                        "marginTop": "15px",
                        "padding": "10px 20px",
                        "backgroundColor": "#4CAF50",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                        "fontSize": "14px",
                        "fontWeight": "bold"
                    }
                ),
                # Store to track editing mode state
                dcc.Store(id="edit-mode-store", data={"edit_mode": True})
            ], style={"display": "flex", "flexDirection": "row"}),

            # Download component for PDF export
            dcc.Download(id="download-pdf")
        ], style={"marginTop": "20px"}),

        # Interval component for checking report status
        dcc.Interval(
            id='report-interval',
            interval=1000,  # in milliseconds
            n_intervals=0,
            disabled=True
        ),

        # Store for encoded images
        dcc.Store(id='image-store'),

        # Store for detection data
        dcc.Store(id='detection-store'),

        # Store for report ID
        dcc.Store(id='report-id')
    ]
)

model = YOLO("yolo11n-tumor-luca.pt")


def _draw_bounding_boxes(img, x1, y1, x2, y2, conf):
    """
    Draw bounding boxes on image and add detection to all_detections
    """
    # Calculate the center of the original bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate the width and height of the original bounding box
    width = x2 - x1
    height = y2 - y1

    # Enlarge the width and height by 25%
    enlarged_width = width * 2
    enlarged_height = height * 2

    # Calculate the new coordinates based on the enlarged dimensions
    # while keeping the same center point
    new_x1 = int(center_x - enlarged_width / 2)
    new_y1 = int(center_y - enlarged_height / 2)
    new_x2 = int(center_x + enlarged_width / 2)
    new_y2 = int(center_y + enlarged_height / 2)

    # Ensure the coordinates stay within the image boundaries
    height, width = img.shape[:2]
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(width - 1, new_x2)
    new_y2 = min(height - 1, new_y2)

    # Draw rectangle - thicker bright green line
    color = (0, 255, 0)  # BGR format - bright green  # 052aff
    cv2.rectangle(img, (new_x1, new_y1), (new_x2, new_y2), color, 8)

    # Add ID with smaller text and better positioning
    label = f"Abnormality {conf:.2f}"

    # Calculate text size to create a background - reduced font size to 1.0 and thickness to 2
    font_size = 2.0
    thickness = 4
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]

    # Draw a filled rectangle as background for text
    # cv2.rectangle(img, (new_x1, new_y1 - text_size[1] - 5), (new_x1 + text_size[0], new_y1), (0, 0, 0), -1)

    # Draw text with bright green - using smaller font size and thickness
    cv2.putText(img, label, (new_x1 - 200, new_y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)


def dicom_to_png_bytes(dicom_bytes, conf_threshold=0.5):
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

        detection_data = {}

        for result in results:
            boxes = result.boxes.xyxy  # [x1, y1, x2, y2] format
            confs = result.boxes.conf  # Confidence scores
            classes = result.boxes.cls  # Class IDs
            print(f"Detected {len(boxes)} objects")

            # Draw bounding boxes with confidence > threshold
            for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                if conf > conf_threshold:
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)

                    # Calculate physical dimensions if pixel spacing is available
                    physical_size = None
                    if hasattr(dcm, "PixelSpacing") and dcm.PixelSpacing is not None:
                        pixel_spacing = dcm.PixelSpacing

                        # Calculate the center of the bounding box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        # Calculate the width and height of the original bounding box
                        width = x2 - x1
                        height = y2 - y1

                        # Enlarge the width and height by 25% for physical size calculation
                        enlarged_width = width * 2
                        enlarged_height = height * 2

                        width_mm = enlarged_width * float(pixel_spacing[0])
                        height_mm = enlarged_height * float(pixel_spacing[1])
                        physical_size = [round(width_mm, 2), round(height_mm, 2)]

                    # Store detection data with original coordinates
                    # The drawing function will handle the enlargement
                    detection_data[str(i)] = {
                        "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                        "physical_size_mm": physical_size,
                        "conf": float(conf),
                        "cls": int(cls),
                        "source": {"file": "current_image"}
                    }

                    _draw_bounding_boxes(pixel_array, x1, y1, x2, y2, conf)

        image = Image.fromarray(pixel_array.astype(np.uint8)).convert("RGB")

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        encoded_png = base64.b64encode(buffered.getvalue()).decode()
        return encoded_png, detection_data
    except Exception as e:
        print(f"Error converting DICOM: {e}")
        return None, None


def generate_mri_report(report_id, stored_images, all_detections, session_memory=None, model_name="o4-mini"):
    """
    Generate MRI report from the uploaded DICOM images
    """
    try:
        if session_memory is None:
            session_memory = []

        if not stored_images:
            report_results[report_id] = {"status": "error", "message": "No images uploaded to generate report."}
            return

        # Process all images and ensure they are in the same order as the display
        image_contents = []

        # The images are already properly sorted at this point (by filename)
        # since we're using the ordered dictionary that was created in process_images
        for image_id, png_encoded in stored_images.items():
            # Store image info
            image_contents.append({
                "filename": image_id,
                "base64": png_encoded
            })

        # Few-shot examples: Negative and Positive reports with proper line spacing
        negative_report = (
            "**Method:** MRI of the neurocranium from 19.03.2025\n\n"
            "**Findings:**\n"
            "Infratentorial regular signaling of both cerebellar hemispheres and the brain stem without evidence of focal lesions. Medium-sized, normally wide 4th ventricle as well as inconspicuous width and configuration of the pericerebellar and perimesencephalic cisterns. Inconspicuous flow void of the basal vessels.\n\n"
            "Symmetrical visualization of the cerebellopontine angle without evidence of a mass.\n"
            "Inconspicuous signaling also of the temporal bones and the mastoids.\n"
            "Supratentorial medium-sized ventricular system of normal width for age without signs of cerebrospinal fluid circulation disorder.\n"
            "Inconspicuous signaling of gray and white matter, no focal lesions. Unobtrusive visualization of the cortical furrow relief.\n\n"
            "**Intracranial vessels supplying the brain:**\n"
            "Inconspicuous vertebral artery on both sides.\n"
            "Inconspicuous basilar artery.\n"
            "Regular posterior cerebral artery on both sides.\n"
            "Inconspicuous ACI in the anterior flow area on both sides with regular carotid T-bifurcation.\n"
            "Inconspicuous anterior cerebral artery.\n"
            "Inconspicuous middle cerebral artery on both sides.\n"
            "No aneurysmal changes. No higher grade stenosis or wall irregularities.\n\n"
            "**Diagnosis:**\n"
            "Inconspicuous visualization of the neurocranium and the intracranial vessels.\n\n"
            "**Yours sincerely**\n"
            "**Your Dr. GPT**"
        )

        positive_report = (
            "**Method:** MRI of the neurocranium from 19.03.2025\n\n"
            "**Findings:**\n"
            "Infratentorial regular signaling of both cerebellar hemispheres and the brain stem without evidence of focal lesions. Medium-sized, normally wide 4th ventricle as well as inconspicuous width and configuration of the pericerebellar and perimesencephalic cisterns. Inconspicuous flow void of the basal vessels.\n\n"
            "Symmetrical visualization of the cerebellopontine angle without evidence of a mass.\n"
            "Inconspicuous signaling also of the temporal bones and the mastoids.\n"
            "Supratentorial medium-sized ventricular system of normal width for age without signs of cerebrospinal fluid circulation disorder.\n"
            "Inconspicuous signaling of gray and white matter, no focal lesions. Unobtrusive visualization of the cortical furrow relief.\n"
            "A 16 x 17 x 10 mm cystic mass of the corpus pineale is identified.\n\n"
            "**Intracranial vessels supplying the brain:**\n"
            "Inconspicuous vertebral artery on both sides.\n"
            "Inconspicuous basilar artery.\n"
            "Regular posterior cerebral artery on both sides.\n"
            "Inconspicuous ACI in the anterior flow area on both sides with regular carotid T-bifurcation.\n"
            "Inconspicuous anterior cerebral artery.\n"
            "Inconspicuous middle cerebral artery on both sides.\n"
            "No aneurysmal changes. No higher grade stenosis or wall irregularities.\n\n"
            "**Diagnosis:**\n"
            "A 16 x 17 x 10 mm cystic mass of the corpus pineale is noted, suggestive of a pineal cyst. No other abnormalities of the neurocranium or intracranial vessels are identified.\n\n"
            "**Yours sincerely**\n"
            "**Your Dr. GPT**"
        )

        # Create a description of the detections for the prompt
        detection_description = ""
        if all_detections:
            detection_description = "Detected objects in the MRI scan:\n"
            for img_id, detections in all_detections.items():
                for id, detection in detections.items():
                    x1, y1, x2, y2 = detection["xyxy"]
                    confidence = detection["conf"]
                    cls = detection["cls"]
                    source_file = detection["source"]["file"] if "source" in detection and "file" in detection[
                        "source"] else "unknown"

                    physical_size_info = ""
                    if "physical_size_mm" in detection and detection["physical_size_mm"]:
                        width_mm, height_mm = detection["physical_size_mm"]
                        physical_size_info = f", physical dimensions: {width_mm}x{height_mm}mm"

                    detection_description += f"- Detection in image {img_id}: In file {source_file}, bounding box coordinates: ({x1}, {y1}, {x2}, {y2}){physical_size_info}, confidence: {confidence:.2f}, class: {cls}\n"

        prompt_prefix = (
            f"You are an assistant that helps generate MRI report templates based on visual observations of pre-processed MRI scans. Today is {datetime.now().strftime('%d.%m.%Y')}.    \n"
            f"I'm providing you with {len(image_contents)} MRI scan images that include bounding boxes from a YOLO model highlighting areas of interest, which may indicate potential abnormalities.\n"
            f"{detection_description}\n"
            "First, describe the visual features of the MRI scans and the bounding boxes (e.g., location, size, shape, contrast, intensity patterns). Include your confidence level in these observations (e.g., high, moderate, low confidence).\n"
            "Then, generate a detailed report template with the following sections: Method, Findings, Intracranial vessels supplying the brain, Diagnosis, and a closing signature ('Yours sincerely, Your Dr. GPT'), based on the visual description. Ensure proper line spacing between paragraphs as shown in the examples.\n"
            "Note: This is not a medical diagnosis; you are only assisting in creating a report template based on visual observations.\n\n"
            "Example 1 (Negative Case):\n"
            f"{negative_report}\n\n"
            "Example 2 (Positive Case):\n"
            f"{positive_report}\n\n"
        )

        # Add previous memory context
        if session_memory:
            last_report = session_memory[-1]
            prompt_prefix += f"Previously generated report:\n{last_report}\n\n"

        # Initialize message content for the model
        message_content = [
            {
                "type": "text",
                "text": prompt_prefix + "Analyze the following MRI scans and generate a report template based on their visual features:"
            }
        ]

        # Add all images to the message content
        for img_info in image_contents:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_info['base64']}",
                    "detail": "auto"
                }
            })

        # Get OpenAI API key
        api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
        if not api_key:
            report_results[report_id] = {"status": "error", "message": "Error: OpenAI API key not found."}
            return

        # Call the OpenAI model
        chat = ChatOpenAI(
            model=model_name,  # Vision-capable model
            api_key=api_key
        )
        with get_openai_callback() as cb:
            result = chat.invoke([HumanMessage(content=message_content)])
            print(cb)
            generated_report = result.content

            # Update session memory
            new_session_memory = session_memory + [generated_report]
            if len(new_session_memory) > 3:
                new_session_memory.pop(0)

            # Store result
            report_results[report_id] = {"status": "complete", "report": generated_report,
                                         "session_memory": new_session_memory}

    except Exception as e:
        print(f"Error generating report: {str(e)}")
        report_results[report_id] = {"status": "error", "message": f"Error generating report: {str(e)}"}


# Background thread to process report generation queue
def report_worker():
    while True:
        try:
            report_task = report_queue.get()
            if report_task is None:
                break

            report_id = report_task["report_id"]
            stored_images = report_task["stored_images"]
            all_detections = report_task["all_detections"]

            # Mark as processing
            report_results[report_id] = {"status": "processing"}

            # Generate report
            # generate_mri_report(report_id, stored_images, all_detections)

        except Exception as e:
            print(f"Error in worker thread: {e}")
        finally:
            report_queue.task_done()


# Start the worker thread
worker_thread = threading.Thread(target=report_worker, daemon=True)
worker_thread.start()


@app.callback(
    [Output('output-images', 'children'),
     Output('image-store', 'data'),
     Output('detection-store', 'data'),
     Output('report-id', 'data'),
     Output('report-output', 'value', allow_duplicate=True),
     Output('report-interval', 'disabled'),
     Output('report-output', 'readOnly', allow_duplicate=True)],
    Input('upload-dicom-folder', 'contents'),
    State('upload-dicom-folder', 'filename'),
    prevent_initial_call=True
)
def process_images(list_of_contents, list_of_names):
    if list_of_contents is not None:
        images = []
        stored_images = OrderedDict()  # Use OrderedDict to maintain insertion order
        all_detections = OrderedDict()  # Use OrderedDict to maintain insertion order

        # Create a list for processing that includes filenames for sorting
        image_data = []

        for i, (content, name) in enumerate(zip(list_of_contents, list_of_names)):
            content_type, content_string = content.split(',')
            dicom_bytes = base64.b64decode(content_string)
            png_encoded, detection_data = dicom_to_png_bytes(dicom_bytes)

            if png_encoded:
                # Store the tuple with all needed data for sorting and processing
                image_data.append((i, name, content, png_encoded, detection_data))

        # Sort the image data by filename
        image_data.sort(key=lambda x: x[1].lower() if x[1] else "")

        # Process the sorted image data
        for i, (original_idx, name, content, png_encoded, detection_data) in enumerate(image_data):
            image_id = f"img-{i}"
            stored_images[image_id] = png_encoded

            # Get a shortened filename if available
            display_name = name if name else f"Image {original_idx + 1}"
            if len(display_name) > 20:
                display_name = display_name[:17] + "..."

            # Store detection data
            if detection_data:
                all_detections[image_id] = detection_data

            # Create a simple button that will trigger the modal
            img_container = html.Div([
                html.Button(
                    id={"type": "image-button", "index": i},
                    children=[
                        html.Img(
                            src=f"data:image/png;base64,{png_encoded}",
                            style={
                                "height": "200px",
                                "width": "auto",
                                "maxWidth": "100%",
                                "borderRadius": "8px"
                            }
                        )
                    ],
                    style={
                        "background": "none",
                        "border": "none",
                        "cursor": "pointer",
                        "padding": "0",
                        "width": "100%"
                    }
                ),
                html.Div(
                    display_name,
                    style={
                        "textAlign": "center",
                        "marginTop": "5px",
                        "fontSize": "14px",
                        "color": "#555",
                        "fontWeight": "bold"
                    }
                ),
                html.Div(
                    "Click to view full size",
                    style={
                        "textAlign": "center",
                        "fontSize": "12px",
                        "color": "#777",
                        "fontStyle": "italic"
                    }
                )
            ],
                style={
                    "display": "inline-block",
                    "marginRight": "15px",
                    "marginBottom": "15px",
                    "width": "220px",
                    "padding": "10px",
                    "backgroundColor": "white",
                    "borderRadius": "10px",
                    "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                    "transition": "transform 0.2s, box-shadow 0.2s",
                    "verticalAlign": "top"
                })

            images.append(img_container)

        # Generate a unique report ID
        report_id = f"report-{time.time()}"

        # Add to the queue for background processing
        report_queue.put({
            "report_id": report_id,
            "stored_images": stored_images,
            "all_detections": all_detections
        })

        # Loading message
        loading_message = "Analyzing images...\n\nPlease wait while our AI analyzes your images. This may take up to a minute."

        return images, stored_images, all_detections, report_id, loading_message, False, True  # True = readOnly during processing

    return no_update, no_update, no_update, no_update, no_update, True, no_update


# Simple callback to show modal when an image button is clicked
@app.callback(
    [Output('image-modal', 'style'),
     Output('modal-image', 'src'),
     Output('current-image-index', 'data'),
     Output('total-images', 'data')],
    [Input({"type": "image-button", "index": ALL}, 'n_clicks')],
    [State('image-store', 'data')],
    prevent_initial_call=True
)
def show_modal(n_clicks, stored_images):
    ctx = callback_context

    if not ctx.triggered or not any(n_clicks):
        raise PreventUpdate

    # Get index of clicked image
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    index = json.loads(trigger_id)['index']
    image_id = f"img-{index}"

    # Get total number of images
    total_images = len(stored_images)

    if image_id in stored_images:
        return {
            "display": "block",
            "position": "fixed",
            "zIndex": "1000",
            "left": "0",
            "top": "0",
            "width": "100%",
            "height": "100%",
            "overflow": "auto",
            "backgroundColor": "rgba(0,0,0,0.9)",
            "textAlign": "center"
        }, f"data:image/png;base64,{stored_images[image_id]}", index, total_images

    raise PreventUpdate


# Callback to navigate to next image
@app.callback(
    [Output('modal-image', 'src', allow_duplicate=True),
     Output('current-image-index', 'data', allow_duplicate=True)],
    [Input('next-image', 'n_clicks'),
     Input('prev-image', 'n_clicks')],
    [State('current-image-index', 'data'),
     State('total-images', 'data'),
     State('image-store', 'data')],
    prevent_initial_call=True
)
def navigate_images(next_clicks, prev_clicks, current_index, total_images, stored_images):
    ctx = callback_context

    if not ctx.triggered:
        raise PreventUpdate

    # Get which button was clicked
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Calculate new index
    new_index = current_index
    if trigger_id == "next-image" and next_clicks:
        new_index = (current_index + 1) % total_images
    elif trigger_id == "prev-image" and prev_clicks:
        new_index = (current_index - 1) % total_images

    # Get image for the new index
    image_id = f"img-{new_index}"

    if image_id in stored_images:
        return f"data:image/png;base64,{stored_images[image_id]}", new_index

    raise PreventUpdate


# Callback to close the modal
@app.callback(
    Output('image-modal', 'style', allow_duplicate=True),
    Input('close-modal', 'n_clicks'),
    prevent_initial_call=True
)
def close_modal(n_clicks):
    if n_clicks:
        return {
            "display": "none",
            "position": "fixed",
            "zIndex": "1000",
            "left": "0",
            "top": "0",
            "width": "100%",
            "height": "100%",
            "overflow": "auto",
            "backgroundColor": "rgba(0,0,0,0.9)",
            "textAlign": "center"
        }

    raise PreventUpdate


@app.callback(
    [Output('report-output', 'value', allow_duplicate=True),
     Output('report-interval', 'disabled', allow_duplicate=True),
     Output('report-output', 'readOnly', allow_duplicate=True)],
    Input('report-interval', 'n_intervals'),
    [State('report-id', 'data'),
     State('edit-mode-store', 'data')],
    prevent_initial_call=True
)
def update_report_status(n_intervals, report_id, edit_mode_data):
    if not report_id or report_id not in report_results:
        return no_update, no_update, no_update

    result = report_results[report_id]
    current_edit_mode = edit_mode_data.get('edit_mode', True)
    read_only = not current_edit_mode  # If in view mode, set readOnly to True

    if result.get("status") == "complete":
        # Report is complete, display it
        report_text = result.get("report", "No report generated")
        return report_text, True, read_only

    elif result.get("status") == "error":
        # There was an error, display error message
        error_message = result.get("message", "An error occurred while generating the report")
        return f"❌ Error\n\n{error_message}", True, read_only

    # Still processing, continue polling (and keep readonly during processing)
    return no_update, no_update, True


# Callback to update the initial report textarea when files are uploaded
@app.callback(
    [Output('report-output', 'value'),
     Output('report-output', 'readOnly')],
    Input('upload-dicom-folder', 'contents'),
    prevent_initial_call=True
)
def reset_report_textarea(contents):
    if contents:
        return "Analyzing images... Please wait for the report to be generated.", True
    return no_update, no_update


# Callback for Export PDF button
@app.callback(
    Output("download-pdf", "data"),
    Input("export-pdf", "n_clicks"),
    State("report-output", "value"),
    prevent_initial_call=True
)
def export_pdf(n_clicks, report_content):
    if n_clicks and report_content:
        # Create a PDF file with the report content
        buffer = io.BytesIO()

        # Create PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()

        # Parse the report content
        elements = []

        # Add title
        title_style = styles["Title"]
        elements.append(Paragraph("MRI Analysis Report", title_style))

        # Add date
        date_style = styles["Normal"]
        date_style.alignment = 1  # Center alignment
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%d.%m.%Y')}", date_style))

        # Add a spacer
        elements.append(Paragraph("<br/><br/>", styles["Normal"]))

        # Process the report content
        for line in report_content.split('\n'):
            if line.startswith('**') and line.endswith('**'):
                # Section headers (marked with **)
                header_text = line.strip('**')
                elements.append(Paragraph(header_text, styles["Heading2"]))
            elif line.strip() == "":
                # Empty lines become spacing
                elements.append(Paragraph("<br/>", styles["Normal"]))
            else:
                # Regular text
                elements.append(Paragraph(line, styles["Normal"]))

        # Build the PDF
        doc.build(elements)

        # Get the value from the BytesIO buffer and encode as base64
        pdf_data = buffer.getvalue()
        buffer.close()

        # Encode the PDF as base64 for Dash Download component
        encoded_pdf = base64.b64encode(pdf_data).decode('utf-8')

        # Create the download data dictionary with base64 content
        filename = f"MRI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return {
            "content": encoded_pdf,
            "filename": filename,
            "type": "application/pdf",
            "base64": True
        }

    return no_update


# Callback to toggle edit mode
@app.callback(
    [Output('report-output', 'readOnly', allow_duplicate=True),
     Output('toggle-edit-mode', 'children'),
     Output('toggle-edit-mode', 'style'),
     Output('edit-mode-store', 'data')],
    Input('toggle-edit-mode', 'n_clicks'),
    State('edit-mode-store', 'data'),
    prevent_initial_call=True
)
def toggle_edit_mode(n_clicks, edit_mode_data):
    if n_clicks:
        # Toggle the edit mode
        current_edit_mode = edit_mode_data.get('edit_mode', True)
        new_edit_mode = not current_edit_mode

        # Update button text and style based on new mode
        if new_edit_mode:
            # Switching to edit mode
            button_text = "Save"
            button_style = {
                "marginTop": "15px",
                "padding": "10px 20px",
                "backgroundColor": "#4CAF50",  # BLUE
                "color": "white",
                "border": "none",
                "borderRadius": "4px",
                "cursor": "pointer",
                "fontSize": "14px",
                "fontWeight": "bold"
            }
            # When edit_mode is True, readOnly should be False
            read_only = False
        else:
            # Switching to view mode
            button_text = "Edit"
            button_style = {
                "marginTop": "15px",
                "padding": "10px 20px",
                "backgroundColor": "#FF9800",  # Orange
                "color": "white",
                "border": "none",
                "borderRadius": "4px",
                "cursor": "pointer",
                "fontSize": "14px",
                "fontWeight": "bold"
            }
            # When edit_mode is False, readOnly should be True
            read_only = True

        return read_only, button_text, button_style, {"edit_mode": new_edit_mode}

    return no_update, no_update, no_update, no_update


# Callback to update the textarea style based on edit mode
@app.callback(
    Output('report-output', 'style'),
    Input('report-output', 'readOnly'),
    prevent_initial_call=True
)
def update_textarea_style(read_only):
    base_style = {
        "width": "100%",
        "height": "400px",
        "padding": "20px",
        "backgroundColor": "white",
        "borderRadius": "10px",
        "boxShadow": "0 4px 8px rgba(0,0,0,0.05)",
        "whiteSpace": "pre-line",
        "fontFamily": "Arial, sans-serif",
        "fontSize": "14px",
        "lineHeight": "1.5",
    }

    if read_only:
        # View mode style
        base_style.update({
            "border": "1px solid #ddd",
            "backgroundColor": "#f9f9f9",
            "cursor": "default"
        })
    else:
        # Edit mode style
        base_style.update({
            "border": "1px solid #4CAF50",
            "backgroundColor": "white",
            "cursor": "text"
        })

    return base_style


if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        # Signal the worker thread to exit
        report_queue.put(None)
        worker_thread.join(timeout=1.0)
