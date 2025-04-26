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
app.title = "Medical Report Generator"

# Define animations using keyframes in a style dictionary
dot_animation_style = {
    "animation": "dot-animation 1.4s infinite",
    "opacity": "0.3"
}

# Update layout to create two pages
app.layout = html.Div(
    style={"backgroundColor": "#f9f9f9", "fontFamily": "Arial, sans-serif", "height": "100vh"},
    children=[
        # Store to track which page to show
        dcc.Store(id="page-state", data={"current_page": "landing"}),
        
        # LANDING PAGE
        html.Div(
            id="landing-page",
            style={"padding": "2rem", "maxWidth": "800px", "margin": "0 auto", "textAlign": "center"},
            children=[
                html.H1("Medical Report Generator", style={"textAlign": "center", "color": "#333", "marginBottom": "10px"}),
                html.P("Upload your DICOM images to detect abnormalities and receive a medical report.", 
                       style={"textAlign": "center", "color": "#666", "marginBottom": "30px"}),
                
                html.Div(
                    style={
                        "border": "2px dashed #ccc",
                        "borderRadius": "10px",
                        "padding": "50px 20px",
                        "backgroundColor": "white",
                        "marginBottom": "20px"
                    },
                    children=[
                        html.Img(src="/assets/upload_icon.png", style={"width": "50px", "height": "50px", "opacity": "0.6", "marginBottom": "20px"}),
                        html.Div("Drop your DICOM folder here", style={"fontSize": "18px", "fontWeight": "bold", "marginBottom": "10px"}),
                        html.Div("or click to browse your files", style={"fontSize": "14px", "color": "#666", "marginBottom": "20px"}),
                        dcc.Upload(
                            id='upload-dicom-folder',
                            children=html.Button("Select Folder",
                                                 style={
                                                     "backgroundColor": "#007bff",
                                                     "color": "white",
                                                     "padding": "10px 20px",
                                                     "border": "none",
                                                     "borderRadius": "5px",
                                                     "cursor": "pointer",
                                                     "fontSize": "14px"
                                                 }),
                            style={
                                "width": "100%",
                                "textAlign": "center",
                            },
                            multiple=True
                        ),
                    ]
                ),
                html.Div("Supported formats: DICOM (.dcm)", style={"fontSize": "12px", "color": "#888"}),
            ]
        ),
        
        # RESULTS PAGE
        html.Div(
            id="results-page",
            style={"display": "none", "padding": "2rem", "maxWidth": "1200px", "margin": "0 auto"},
            children=[
                # Top navigation bar with back button
                html.Div(
                    style={"display": "flex", "alignItems": "center", "marginBottom": "20px"},
                    children=[
                        html.Button(
                            "← Back", 
                            id="back-button",
                            style={
                                "backgroundColor": "transparent",
                                "border": "none",
                                "color": "#007bff",
                                "cursor": "pointer",
                                "fontSize": "14px",
                                "marginRight": "20px"
                            }
                        ),
                        html.H2("Medical Report", style={"margin": "0", "flexGrow": "1"}),
                    ]
                ),
                
                # Main content - two columns
                html.Div(
                    style={"display": "flex", "gap": "20px"},
                    children=[
                        # Left column - Report
                        html.Div(
                            style={
                                "flex": "1", 
                                "backgroundColor": "white", 
                                "borderRadius": "10px", 
                                "padding": "0",
                                "boxShadow": "0 2px 5px rgba(0,0,0,0.1)",
                                "display": "flex",
                                "flexDirection": "column"
                            },
                            children=[
                                dcc.Textarea(
                                    id="report-output",
                                    style={
                                        "width": "100%",
                                        "height": "calc(100vh - 250px)",
                                        "padding": "20px",
                                        "backgroundColor": "white",
                                        "borderRadius": "10px",
                                        "whiteSpace": "pre-line",
                                        "fontFamily": "Arial, sans-serif",
                                        "fontSize": "14px",
                                        "lineHeight": "1.5",
                                        "marginBottom": "15px",
                                        "border": "none",
                                        "resize": "none",
                                        "boxSizing": "border-box"
                                    },
                                    value="Analyzing images...\n\nPlease wait while our AI analyzes your images. This may take up to a minute.",
                                    readOnly=True
                                ),
                                html.Div(
                                    style={
                                        "display": "flex", 
                                        "justifyContent": "flex-end", 
                                        "gap": "10px",
                                        "padding": "0 20px 20px 20px"
                                    },
                                    children=[
                                        html.Button(
                                            "Edit", 
                                            id="toggle-edit-mode",
                                            style={
                                                "padding": "8px 20px",
                                                "backgroundColor": "#FF9800",
                                                "color": "white",
                                                "border": "none",
                                                "borderRadius": "4px",
                                                "cursor": "pointer",
                                                "fontSize": "14px"
                                            }
                                        ),
                                        html.Button(
                                            "Download PDF", 
                                            id="export-pdf",
                                            style={
                                                "padding": "8px 20px",
                                                "backgroundColor": "#2196F3",
                                                "color": "white",
                                                "border": "none",
                                                "borderRadius": "4px",
                                                "cursor": "pointer",
                                                "fontSize": "14px"
                                            }
                                        ),
                                    ]
                                )
                            ]
                        ),
                        
                        # Right column - Images
                        html.Div(
                            style={"flex": "1", "backgroundColor": "white", "borderRadius": "10px", "padding": "20px", "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"},
                            children=[
                                html.H3("Images", style={"marginTop": "0", "marginBottom": "15px"}),
                                
                                # Navigation controls
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "justifyContent": "center",
                                        "alignItems": "center",
                                        "marginBottom": "15px",
                                        "gap": "15px"
                                    },
                                    children=[
                                        html.Button(
                                            "▲", 
                                            id="prev-display-image",
                                            style={
                                                "fontSize": "16px",
                                                "padding": "5px 15px",
                                                "backgroundColor": "#f0f0f0",
                                                "border": "none",
                                                "borderRadius": "4px",
                                                "cursor": "pointer"
                                            }
                                        ),
                                        html.Div(
                                            id="image-counter",
                                            style={"fontSize": "14px", "fontWeight": "bold"},
                                            children="Image 1 of 1"
                                        ),
                                        html.Button(
                                            "▼", 
                                            id="next-display-image",
                                            style={
                                                "fontSize": "16px",
                                                "padding": "5px 15px",
                                                "backgroundColor": "#f0f0f0",
                                                "border": "none",
                                                "borderRadius": "4px",
                                                "cursor": "pointer"
                                            }
                                        ),
                                    ]
                                ),
                                
                                # Single image display
                                html.Div(
                                    id="current-image-display",
                                    style={
                                        "display": "flex",
                                        "justifyContent": "center",
                                        "alignItems": "center",
                                        "height": "calc(100vh - 300px)",
                                        "backgroundColor": "#f9f9f9",
                                        "borderRadius": "8px",
                                        "overflow": "hidden"
                                    }
                                ),
                                
                                # Hide the output-images div but keep it for storing all images
                                html.Div(
                                    id="output-images",
                                    style={"display": "none"}
                                )
                            ]
                        )
                    ]
                ),
                
                # Download component for PDF export
                dcc.Download(id="download-pdf")
            ]
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
        dcc.Store(id='report-id'),
        
        # Store to track editing mode state
        dcc.Store(id="edit-mode-store", data={"edit_mode": False}),
        
        # Store for current displayed image index
        dcc.Store(id='current-display-image-index', data=0)
    ]
)

model = YOLO("yolo11n-tumor-luca.pt")


def _draw_bounding_boxes(img, x1, y1, x2, y2):
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
    enlarged_width = width * 1.25
    enlarged_height = height * 1.25
    
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
    GREEN = (0, 255, 0)  # BGR format - bright green
    cv2.rectangle(img, (new_x1, new_y1), (new_x2, new_y2), GREEN, 3)

    # Add ID with smaller text and better positioning
    label = f"Abnormality"

    # Calculate text size to create a background - reduced font size to 1.0 and thickness to 2
    font_size = 1.0
    thickness = 2
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]

    # Draw a filled rectangle as background for text
    cv2.rectangle(img,
                  (new_x1, new_y1 - text_size[1] - 5),
                  (new_x1 + text_size[0], new_y1),
                  (0, 0, 0),
                  -1)  # -1 means filled

    # Draw text with bright green - using smaller font size and thickness
    cv2.putText(img, label, (new_x1, new_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, GREEN, thickness)


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
                        enlarged_width = width * 1.25
                        enlarged_height = height * 1.25
                        
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

                    _draw_bounding_boxes(pixel_array, x1, y1, x2, y2)

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
                    source_file = detection["source"]["file"] if "source" in detection and "file" in detection["source"] else "unknown"
                    
                    physical_size_info = ""
                    if "physical_size_mm" in detection and detection["physical_size_mm"]:
                        width_mm, height_mm = detection["physical_size_mm"]
                        physical_size_info = f", physical dimensions: {width_mm}x{height_mm}mm"
                        
                    detection_description += f"- Detection in image {img_id}: In file {source_file}, bounding box coordinates: ({x1}, {y1}, {x2}, {y2}){physical_size_info}, confidence: {confidence:.2f}, class: {cls}\n"

        prompt_prefix = (
            f"You are an assistant that helps generate MRI report templates based on visual observations of pre-processed MRI scans. Today is {datetime.now().strftime('%d.%m.%Y')}.    \n"
            f"I'm providing you with {len(image_contents)} MRI scan images that include bounding boxes from a YOLO model highlighting areas of interest, which may indicate potential abnormalities.\n"
            f"{detection_description}\n"
            "First, describe the visual features of the MRI scans and the bounding boxes (e.g., location, size, shape, contrast, intensity patterns) and started with Explaination. Include your confidence level in these observations (e.g., high, moderate, low confidence).\n"
            "Then, generate a detailed report template started with 'MRI Report Template' and have the following sections: Method, Findings, Intracranial vessels supplying the brain, Diagnosis, and a closing signature ('Yours sincerely, Your Dr. GPT'), based on the visual description. Ensure proper line spacing between paragraphs as shown in the examples.\n"
            "Note: This is not a medical diagnosis; you are only assisting in creating a report template based on visual observations. \n\n"
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
            report_results[report_id] = {"status": "complete", "report": generated_report, "session_memory": new_session_memory}

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
            generate_mri_report(report_id, stored_images, all_detections)
            
        except Exception as e:
            print(f"Error in worker thread: {e}")
        finally:
            report_queue.task_done()


# Start the worker thread
worker_thread = threading.Thread(target=report_worker, daemon=True)
worker_thread.start()


# New callback to switch pages when DICOM files are uploaded
@app.callback(
    [Output('landing-page', 'style'),
     Output('results-page', 'style'),
     Output('page-state', 'data')],
    [Input('upload-dicom-folder', 'contents'),
     Input('back-button', 'n_clicks')],
    [State('page-state', 'data')],
    prevent_initial_call=True
)
def toggle_page_visibility(upload_contents, back_button, page_state):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Default styles
    landing_style = {"padding": "2rem", "maxWidth": "800px", "margin": "0 auto", "textAlign": "center"}
    results_style = {"display": "none", "padding": "2rem", "maxWidth": "1200px", "margin": "0 auto"}
    
    if trigger_id == 'upload-dicom-folder' and upload_contents:
        # User uploaded files, show results page
        landing_style["display"] = "none"
        results_style["display"] = "block"
        new_page_state = {"current_page": "results"}
    elif trigger_id == 'back-button' and back_button:
        # User clicked back, show landing page
        landing_style["display"] = "block"
        results_style["display"] = "none"
        new_page_state = {"current_page": "landing"}
    else:
        # Keep current state
        return no_update, no_update, no_update
        
    return landing_style, results_style, new_page_state


# Modify the process_images callback to update for new layout
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
            display_name = name if name else f"Image {original_idx+1}"
            if len(display_name) > 20:
                display_name = display_name[:17] + "..."
            
            # Store detection data
            if detection_data:
                all_detections[image_id] = detection_data
            
            # Create a simple button that will trigger the modal - updated styling for the new layout
            img_container = html.Div([
                html.Button(
                    id={"type": "image-button", "index": i},
                    children=[
                        html.Img(
                            src=f"data:image/png;base64,{png_encoded}",
                            style={
                                "width": "100%", 
                                "height": "auto",
                                "borderRadius": "8px",
                                "objectFit": "contain"
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
                )
            ],
            style={
                "marginBottom": "15px",
                "backgroundColor": "#f5f5f5",
                "borderRadius": "10px",
                "padding": "10px",
                "transition": "transform 0.2s, box-shadow 0.2s",
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


# Update the initialize_image_display callback to use pattern-matching IDs
@app.callback(
    [Output('current-image-display', 'children', allow_duplicate=True),
     Output('image-counter', 'children', allow_duplicate=True),
     Output('current-display-image-index', 'data', allow_duplicate=True)],
    Input('image-store', 'data'),
    prevent_initial_call=True
)
def initialize_image_display(stored_images):
    if not stored_images:
        return html.Div("No images uploaded"), "Image 0 of 0", 0
        
    # Convert stored_images to a list for easier indexing
    img_list = list(stored_images.items())
    total_images = len(img_list)
    
    if total_images > 0:
        image_id, png_encoded = img_list[0]
        # Wrap the image in a button with a pattern-matching ID
        img = html.Button(
            id={"type": "main-image-button", "index": 0},
            children=[
                html.Img(
                    src=f"data:image/png;base64,{png_encoded}",
                    style={
                        "maxWidth": "100%",
                        "maxHeight": "100%",
                        "objectFit": "contain"
                    }
                )
            ],
            style={
                "background": "none",
                "border": "none",
                "cursor": "pointer",
                "padding": "0",
                "width": "100%",
                "height": "100%",
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center"
            }
        )
        counter_text = f"Image 1 of {total_images}"
        return img, counter_text, 0
    
    return html.Div("No images uploaded"), "Image 0 of 0", 0


# Update the navigation callback to use pattern-matching IDs
@app.callback(
    [Output('current-image-display', 'children'),
     Output('image-counter', 'children'),
     Output('current-display-image-index', 'data')],
    [Input('prev-display-image', 'n_clicks'),
     Input('next-display-image', 'n_clicks')],
    [State('image-store', 'data'),
     State('current-display-image-index', 'data')],
    prevent_initial_call=True
)
def navigate_display_images(prev_clicks, next_clicks, stored_images, current_index):
    if not stored_images:
        return html.Div("No images uploaded"), "Image 0 of 0", 0
        
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
        
    # Convert stored_images to a list for easier indexing
    img_list = list(stored_images.items())
    total_images = len(img_list)
    
    if total_images == 0:
        return html.Div("No images uploaded"), "Image 0 of 0", 0
    
    # Get which button was clicked
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Calculate new index
    new_index = current_index
    if trigger_id == "next-display-image" and next_clicks:
        new_index = (current_index + 1) % total_images
    elif trigger_id == "prev-display-image" and prev_clicks:
        new_index = (current_index - 1) % total_images
    
    # Get image for the new index
    if total_images > 0:
        image_id, png_encoded = img_list[new_index]
        # Wrap the image in a button with a pattern-matching ID
        img = html.Button(
            id={"type": "main-image-button", "index": new_index},
            children=[
                html.Img(
                    src=f"data:image/png;base64,{png_encoded}",
                    style={
                        "maxWidth": "100%",
                        "maxHeight": "100%",
                        "objectFit": "contain"
                    }
                )
            ],
            style={
                "background": "none",
                "border": "none",
                "cursor": "pointer",
                "padding": "0",
                "width": "100%",
                "height": "100%",
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center"
            }
        )
        counter_text = f"Image {new_index + 1} of {total_images}"
        return img, counter_text, new_index
    
    return html.Div("No images uploaded"), "Image 0 of 0", 0


# Update the modal callback to use pattern-matching IDs for the main image button
@app.callback(
    [Output('image-modal', 'style'),
     Output('modal-image', 'src'),
     Output('current-image-index', 'data'),
     Output('total-images', 'data')],
    [Input({"type": "image-button", "index": ALL}, 'n_clicks'),
     Input({"type": "main-image-button", "index": ALL}, 'n_clicks')],
    [State('image-store', 'data'),
     State('current-display-image-index', 'data')],
    prevent_initial_call=True
)
def show_modal(thumbnail_clicks, main_image_clicks, stored_images, current_display_index):
    ctx = callback_context
    
    if not ctx.triggered or (not any(thumbnail_clicks) and not any(main_image_clicks)):
        raise PreventUpdate
    
    # Get total number of images
    total_images = len(stored_images) if stored_images else 0
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Check if the click came from the main displayed image
    if '"type":"main-image-button"' in trigger_id and any(main_image_clicks):
        # Clicked on the main displayed image
        img_list = list(stored_images.items())
        if current_display_index < len(img_list):
            image_id, png_encoded = img_list[current_display_index]
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
            }, f"data:image/png;base64,{png_encoded}", current_display_index, total_images
    else:
        # Handle original thumbnail clicks
        try:
            # Get index of clicked image from thumbnail grid
            index = json.loads(trigger_id)['index']
            image_id = f"img-{index}"
            
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
        except:
            # Handle any parsing errors
            pass
    
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
    current_edit_mode = edit_mode_data.get('edit_mode', False)
    read_only = not current_edit_mode  # If in edit mode, set readOnly to False
    
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
        
        # Add elements to build PDF
        elements = []
        
        # Add title
        title_style = styles["Title"]
        elements.append(Paragraph("Medical Report", title_style))
        
        # Add date
        date_style = styles["Normal"]
        date_style.alignment = 1  # Center alignment
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%d.%m.%Y')}", date_style))
        
        # Add a spacer
        elements.append(Paragraph("<br/><br/>", styles["Normal"]))
        
        # Process the report content by sections based on markdown formatting
        current_section = None
        section_content = []
        
        for line in report_content.split('\n'):
            # Process headings (lines that start with # or ##)
            if line.startswith('# '):
                # If we have content from a previous section, add it first
                if current_section and section_content:
                    header_style = styles["Heading2"]
                    elements.append(Paragraph(current_section, header_style))
                    
                    # Add the content as paragraphs
                    for content_line in section_content:
                        elements.append(Paragraph(content_line, styles["Normal"]))
                    
                    # Add spacing after section
                    elements.append(Paragraph("<br/>", styles["Normal"]))
                
                # Start a new main section
                current_section = line[2:].strip()  # Remove the # and spaces
                section_content = []
                
            elif line.startswith('## '):
                # If we have content from a previous section, add it first
                if current_section and section_content:
                    header_style = styles["Heading2"]
                    elements.append(Paragraph(current_section, header_style))
                    
                    # Add the content as paragraphs
                    for content_line in section_content:
                        elements.append(Paragraph(content_line, styles["Normal"]))
                    
                    # Add spacing after section
                    elements.append(Paragraph("<br/>", styles["Normal"]))
                
                # Start a new subsection
                current_section = line[3:].strip()  # Remove the ## and spaces
                section_content = []
                
            elif line.strip() == "":
                # Empty line - add spacing if we have content
                if section_content:
                    section_content.append("")
            else:
                # Regular text line - add to current section content
                section_content.append(line)
        
        # Add the last section if there is one
        if current_section and section_content:
            header_style = styles["Heading2"]
            elements.append(Paragraph(current_section, header_style))
            
            # Add the content as paragraphs
            for content_line in section_content:
                if content_line:  # Skip empty strings
                    elements.append(Paragraph(content_line, styles["Normal"]))
        
        # Build the PDF
        doc.build(elements)
        
        # Get the value from the BytesIO buffer and encode as base64
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Encode the PDF as base64 for Dash Download component
        encoded_pdf = base64.b64encode(pdf_data).decode('utf-8')
        
        # Create the download data dictionary with base64 content
        filename = f"Medical_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
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
        current_edit_mode = edit_mode_data.get('edit_mode', False)
        new_edit_mode = not current_edit_mode
        
        # Update button text and style based on new mode
        if new_edit_mode:
            # Switching to edit mode
            button_text = "Save"
            button_style = {
                "padding": "8px 20px",
                "backgroundColor": "#4CAF50",  # Green
                "color": "white",
                "border": "none",
                "borderRadius": "4px",
                "cursor": "pointer",
                "fontSize": "14px"
            }
            # When edit_mode is True, readOnly should be False
            read_only = False
        else:
            # Switching to view mode
            button_text = "Edit"
            button_style = {
                "padding": "8px 20px",
                "backgroundColor": "#FF9800",  # Orange
                "color": "white",
                "border": "none",
                "borderRadius": "4px",
                "cursor": "pointer",
                "fontSize": "14px"
            }
            # When edit_mode is False, readOnly should be True
            read_only = True
        
        return read_only, button_text, button_style, {"edit_mode": new_edit_mode}
    
    return no_update, no_update, no_update, no_update


# Update the textarea style based on edit mode
@app.callback(
    Output('report-output', 'style'),
    Input('report-output', 'readOnly'),
    prevent_initial_call=True
)
def update_textarea_style(read_only):
    base_style = {
        "width": "100%",
        "height": "calc(100vh - 250px)",
        "padding": "20px",
        "backgroundColor": "white",
        "borderRadius": "10px",
        "whiteSpace": "pre-line",
        "fontFamily": "Arial, sans-serif",
        "fontSize": "14px",
        "lineHeight": "1.5",
        "marginBottom": "15px",
        "border": "none",
        "resize": "none",
        "boxSizing": "border-box"
    }
    
    if read_only:
        # View mode style
        base_style.update({
            "backgroundColor": "#f9f9f9",
            "cursor": "default"
        })
    else:
        # Edit mode style
        base_style.update({
            "backgroundColor": "white",
            "cursor": "text",
            "outline": "1px solid #4CAF50" # Use outline instead of border
        })
    
    return base_style


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


# Callback to navigate within the modal
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
def navigate_modal_images(next_clicks, prev_clicks, current_index, total_images, stored_images):
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
    img_list = list(stored_images.items())
    if 0 <= new_index < len(img_list):
        image_id, png_encoded = img_list[new_index]
        return f"data:image/png;base64,{png_encoded}", new_index
    
    raise PreventUpdate


if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        # Signal the worker thread to exit
        report_queue.put(None)
        worker_thread.join(timeout=1.0)
