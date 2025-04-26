from os import listdir, makedirs
from os.path import join, isfile, exists
import time
import cv2
import numpy as np
import json
import base64
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_community.callbacks.manager import get_openai_callback
import pydicom
from pathlib import Path

from ultralytics import YOLO

# Custom JSON encoder to handle pydicom MultiValue and other non-serializable types
class DicomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, list, dict)):
            return list(obj)
        # Handle MultiValue from pydicom
        if hasattr(obj, 'original_string'):
            return str(obj)
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle numpy data types
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        # Handle bytes
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        # Handle unknown types
        try:
            return str(obj)
        except:
            return "UNSERIALIZABLE OBJECT"

def _read_dicom_file(dicom_path):
    """
    Read a DICOM file and return its pixel array and metadata
    """
    try:
        dicom = pydicom.dcmread(dicom_path)
        
        # Extract metadata
        metadata = {
            "PatientID": dicom.PatientID if hasattr(dicom, "PatientID") else "Unknown",
            "PatientName": str(dicom.PatientName) if hasattr(dicom, "PatientName") else "Unknown",
            "PatientAge": dicom.PatientAge if hasattr(dicom, "PatientAge") else "Unknown",
            "PatientSex": dicom.PatientSex if hasattr(dicom, "PatientSex") else "Unknown",
            "StudyDate": dicom.StudyDate if hasattr(dicom, "StudyDate") else "Unknown",
            "Modality": dicom.Modality if hasattr(dicom, "Modality") else "Unknown",
            "PixelSpacing": dicom.PixelSpacing if hasattr(dicom, "PixelSpacing") else None,
            "SliceThickness": dicom.SliceThickness if hasattr(dicom, "SliceThickness") else None,
            "ImageOrientationPatient": dicom.ImageOrientationPatient if hasattr(dicom, "ImageOrientationPatient") else None,
        }
        
        # Convert to pixel array
        pixel_array = dicom.pixel_array
        
        # Normalize to 8-bit for display
        if pixel_array.dtype != np.uint8:
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        return pixel_array, metadata
    
    except Exception as e:
        print(f"Error reading DICOM file {dicom_path}: {str(e)}")
        return None, None

def _read_dicom_folder(image_dir, output_dir, f):
    """
    Read a file (DICOM or regular) and return detection path and image
    """
    file_path = join(image_dir, f)
    file_ext = Path(f).suffix.lower()
    
    # Process DICOM files
    if file_ext == '.dcm':
        img_array, metadata = _read_dicom_file(file_path)
        if img_array is None:
            return None, None, None
            
        # Create a temporary JPG file for YOLO detection
        temp_jpg_path = join(output_dir, f"temp_{Path(f).stem}.jpg")
        
        # Resize image to 480x480
        img_resized = cv2.resize(img_array, (480, 480))
        
        # Convert to BGR to ensure colors display correctly (if the image is grayscale)
        if len(img_resized.shape) == 2:  # If grayscale
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            
        cv2.imwrite(temp_jpg_path, img_resized)
        
        # Use the temp JPG for detection
        detection_path = temp_jpg_path
        img = img_resized.copy()
        return detection_path, img, metadata
    elif file_ext == '.DS_Store' or file_ext == '.json' or file_ext == '.DS_Store':
        return None, None, None
    else:
        return None, None, None

def _draw_bounding_boxes(img, x1, y1, x2, y2, id, detection_data, all_detections):
    """
    Draw bounding boxes on image and add detection to all_detections
    """
    # Add to all_detections with ID as key
    all_detections[str(id)] = detection_data
    
    # Draw rectangle - thicker bright green line
    GREEN = (0, 255, 0)  # BGR format - bright green
    cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 3)
    
    # Add ID with smaller text and better positioning
    label = f"ID {id}"
    
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
    cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, font_size, GREEN, thickness)

def label_image(model, image_dir, output_dir, conf_threshold=0.5):
    """
    Process images (DICOM or regular) and apply the model for object detection.
    """
    files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]
    id = 0
    
    # Dictionary to store all detections with ID as key
    all_detections = {}
    # Dictionary to store metadata for each processed file
    all_metadata = {}
    
    for f in files:
        result = _read_dicom_folder(image_dir, output_dir, f)
        detection_path, img, metadata = result
        
        if detection_path is None:
            continue
            
        # Store metadata
        all_metadata[f] = metadata
        
        # Run object detection
        results = model(detection_path)
        
        for result in results:
            boxes = result.boxes.xyxy  # [x1, y1, x2, y2] format
            confs = result.boxes.conf  # Confidence scores
            classes = result.boxes.cls  # Class IDs
            print(f"Detected {len(boxes)} objects in {f}")
            
            # Draw bounding boxes with confidence > threshold
            for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                if conf > conf_threshold:
                    id += 1
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    
                    # Calculate physical dimensions if pixel spacing is available
                    physical_size = None
                    if all_metadata[f]["PixelSpacing"] is not None:
                        pixel_spacing = all_metadata[f]["PixelSpacing"]
                        width_mm = (x2 - x1) * float(pixel_spacing[0])
                        height_mm = (y2 - y1) * float(pixel_spacing[1])
                        physical_size = [round(width_mm, 2), round(height_mm, 2)]
                    
                    # Store detection data
                    detection_data = {
                        "id": id,
                        "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                        "physical_size_mm": physical_size,
                        "conf": float(conf),
                        "cls": int(cls),
                        "source": {
                            "file": f,
                            "path": join(image_dir, f)
                        }
                        # "metadata": all_metadata[f]
                    }
                    
                    _draw_bounding_boxes(img, x1, y1, x2, y2, id, detection_data, all_detections)
            
            # Save the annotated image
            output_path = join(output_dir, f"annotated_{Path(f).stem}.jpg")
            cv2.imwrite(output_path, img)
            
            # Clean up temporary file if needed
            if Path(f).suffix.lower() == '.dcm':
                temp_jpg_path = join(output_dir, f"temp_{Path(f).stem}.jpg")
                if exists(temp_jpg_path):
                    import os
                    os.remove(temp_jpg_path)
    
    # Save detection results as JSON
    json_path = join(output_dir, "detections.json")
    with open(json_path, 'w') as json_file:
        json.dump(all_detections, json_file, indent=4, cls=DicomJSONEncoder)
    
    # Save metadata separately
    metadata_path = join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as metadata_file:
        json.dump(all_metadata, metadata_file, indent=4, cls=DicomJSONEncoder)
    
    print(f"Detection results saved to {json_path}")
    print(f"Metadata saved to {metadata_path}")
    return all_detections, all_metadata

def generate_mri_report(label_image_folder, all_detections, session_memory=None, model="o4-mini", metadata=None):
    if session_memory is None:
        session_memory = []
    
    # Find all annotated images
    annotated_files = [f for f in listdir(label_image_folder) if f.startswith("annotated_")]
    if not annotated_files:
        return "No annotated images found to generate report.", session_memory
    
    # Process all images
    image_contents = []
    for img_file in annotated_files:
        image_path = join(label_image_folder, img_file)
        
        # Convert each image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Store image info
            image_contents.append({
                "filename": img_file,
                "base64": base64_image
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
        for id, detection in all_detections.items():
            x1, y1, x2, y2 = detection["xyxy"]
            confidence = detection["conf"]
            cls = detection["cls"]
            source_file = detection["source"]["file"] if "source" in detection and "file" in detection["source"] else "unknown"
            
            physical_size_info = ""
            if "physical_size_mm" in detection and detection["physical_size_mm"]:
                width_mm, height_mm = detection["physical_size_mm"]
                physical_size_info = f", physical dimensions: {width_mm}x{height_mm}mm"
                
            detection_description += f"- Detection #{id}: In file {source_file}, bounding box coordinates: ({x1}, {y1}, {x2}, {y2}){physical_size_info}, confidence: {confidence:.2f}, class: {cls}\n"
    
    # Add metadata information
    metadata_description = ""
    if metadata:
        metadata_description = "\nMetadata from DICOM images:\n"
        for filename, file_metadata in metadata.items():
            metadata_description += f"- File: {filename}\n"
            if "PatientID" in file_metadata:
                metadata_description += f"  - Patient ID: {file_metadata['PatientID']}\n"
            if "StudyDate" in file_metadata:
                metadata_description += f"  - Study Date: {file_metadata['StudyDate']}\n"
            if "Modality" in file_metadata:
                metadata_description += f"  - Modality: {file_metadata['Modality']}\n"
            if "SliceThickness" in file_metadata and file_metadata["SliceThickness"]:
                metadata_description += f"  - Slice Thickness: {file_metadata['SliceThickness']} mm\n"

    prompt_prefix = (
        "You are an assistant that helps generate MRI report templates based on visual observations of pre-processed MRI scans.\n"
        f"I'm providing you with {len(image_contents)} MRI scan images that include bounding boxes from a YOLO model highlighting areas of interest, which may indicate potential aneurysms or other abnormalities.\n"
        f"{detection_description}\n"
        # f"{metadata_description}\n"
        "First, describe the visual features of the MRI scans and the bounding boxes (e.g., location, size, shape, contrast, intensity patterns). Include your confidence level in these observations (e.g., high, moderate, low confidence).\n"
        "Then, generate a detailed report template with the following sections: Method, Findings, Intracranial vessels supplying the brain, Diagnosis, and a closing signature ('Yours sincerely, Your Dr. GPT'), based on the visual description. Ensure proper line spacing between paragraphs as shown in the examples.\n"
        "Note: This is not a medical diagnosis; you are only assisting in creating a report template based on visual observations.\n\n"
        "Example 1 (Negative Case):\n"
        f"{negative_report}\n\n"
        "Example 2 (Positive Case):\n"
        f"{positive_report}\n\n"
        "Not please doesn't mention about the bounding boxes, just describe the MRI scan and the findings and the related images."
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

    # Call the Open AI model
    try:
        chat = ChatOpenAI(
            model=model,  # Vision-capable model
            # max_tokens=1024,
            api_key="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"
        )
        with get_openai_callback() as cb:
            result = chat.invoke([HumanMessage(content=message_content)])
            print(cb)
            generated_report = result.content

            # Update session memory
            new_session_memory = session_memory + [generated_report]
            if len(new_session_memory) > 3:
                new_session_memory.pop(0)

            return generated_report, new_session_memory

    except Exception as e:
        return f"Error generating report: {str(e)}", session_memory

def main():

    # image_dir = "brain-tumor/valid/images"
    # image_dir = "luca/Series-018 sag"

    yolo_model = YOLO("yolo11n-tumor-luca.pt")
    image_dir = "/Users/hoaquinng/Desktop/ScanSquad/updated_dicoms"
    output_dir = f"/Users/hoaquinng/Desktop/ScanSquad/test_output_images/{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directory if it doesn't exist
    if not exists(output_dir):
        makedirs(output_dir)
    
    # Save annotated images into output_dir and get all detections details.
    all_detections, all_metadata = label_image(yolo_model, image_dir, output_dir)
    
    # Generate MRI report using the first image
    report, session_memory = generate_mri_report(output_dir, all_detections, model="o4-mini", metadata=all_metadata)
    print("\nGenerated MRI Report:")
    print(report)

if __name__ == '__main__':
    main()
