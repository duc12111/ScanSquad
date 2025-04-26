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
                
                # Simple image container
                html.Img(
                    id="modal-image",
                    style={
                        "maxWidth": "90%",
                        "maxHeight": "90%",
                        "margin": "40px auto",
                        "display": "block"
                    }
                )
            ]
        ),
        
        # Report output area with loading indicator
        html.Div([
            html.H2("MRI Analysis Report", style={"color": "#333", "marginBottom": "10px"}),
            html.Div(
                id="report-output",
                style={
                    "minHeight": "200px",
                    "padding": "20px",
                    "backgroundColor": "white",
                    "borderRadius": "10px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.05)",
                    "whiteSpace": "pre-line"
                },
                children="Upload images to generate a report"
            )
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
        
        detection_data = {}

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
            "You are an assistant that helps generate MRI report templates based on visual observations of pre-processed MRI scans.\n"
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


@app.callback(
    [Output('output-images', 'children'),
     Output('image-store', 'data'),
     Output('detection-store', 'data'),
     Output('report-id', 'data'),
     Output('report-output', 'children'),
     Output('report-interval', 'disabled')],
    Input('upload-dicom-folder', 'contents'),
    State('upload-dicom-folder', 'filename')
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
        
        # Show loading message with animation
        loading_message = html.Div([
            html.Div([
                html.Span("Analyzing images...", style={"fontSize": "18px", "fontWeight": "bold"}),
            ]),
            html.Div([
                html.Div([
                    html.Span("●", style={"color": "#333", "opacity": "0.3", "animationName": "pulse", "animationDuration": "1.4s", "animationIterationCount": "infinite", "animationDelay": "0s"}),
                    html.Span("●", style={"color": "#333", "opacity": "0.3", "animationName": "pulse", "animationDuration": "1.4s", "animationIterationCount": "infinite", "animationDelay": "0.2s"}),
                    html.Span("●", style={"color": "#333", "opacity": "0.3", "animationName": "pulse", "animationDuration": "1.4s", "animationIterationCount": "infinite", "animationDelay": "0.4s"}),
                ], style={"marginTop": "10px"})
            ]),
            html.Div([
                html.Span("Please wait while our AI analyzes your images. This may take up to a minute.",
                     style={"marginTop": "20px", "color": "#666"})
            ], style={"marginTop": "20px"})
        ])
        
        return images, stored_images, all_detections, report_id, loading_message, False
    
    return no_update, no_update, no_update, no_update, no_update, True


# Simple callback to show modal when an image button is clicked
@app.callback(
    [Output('image-modal', 'style'),
     Output('modal-image', 'src')],
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
        }, f"data:image/png;base64,{stored_images[image_id]}"
    
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
    [Output('report-output', 'children', allow_duplicate=True),
     Output('report-interval', 'disabled', allow_duplicate=True)],
    Input('report-interval', 'n_intervals'),
    State('report-id', 'data'),
    prevent_initial_call=True
)
def update_report_status(n_intervals, report_id):
    if not report_id or report_id not in report_results:
        return no_update, no_update
    
    result = report_results[report_id]
    
    if result.get("status") == "complete":
        # Report is complete, display it
        report_text = result.get("report", "No report generated")
        return dcc.Markdown(report_text), True
    
    elif result.get("status") == "error":
        # There was an error, display error message
        error_message = result.get("message", "An error occurred while generating the report")
        return html.Div([
            html.Div("❌ Error", style={"color": "red", "fontWeight": "bold", "fontSize": "18px"}),
            html.Div(error_message, style={"marginTop": "10px"})
        ]), True
    
    # Still processing, continue polling
    return no_update, no_update


if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        # Signal the worker thread to exit
        report_queue.put(None)
        worker_thread.join(timeout=1.0)
