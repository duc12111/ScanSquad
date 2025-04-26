# ScanSquad

## MRI Aneurysm Report Generator

### Overview
This project, developed by ScanSquad, is a Dash web application that generates MRI report templates for brain scans. It processes pre-processed MRI images with YOLO bounding boxes indicating potential aneurysms or other abnormalities. Using LangChain and Open AI's `gpt-4o` model, the app describes visual features and generates structured reports in a specified format, avoiding medical diagnosis to comply with Open AI's restrictions.

### Features
- Upload MRI scans (JPEG/PNG) with YOLO bounding boxes.
- Generate detailed report templates with sections: Method, Findings, Intracranial vessels supplying the brain, Diagnosis, and a closing signature.
- Includes few-shot examples for both positive and negative cases to guide report generation.
- Maintains session memory to provide context from previous reports.

### Installation
1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then install the required packages:
   ```
   pip install dash dash-bootstrap-components langchain langchain-openai langchain-community python-dotenv
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the project directory and add your Open AI API key:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

### Usage
1. **Run the Application**:
   Save the code as `mri_report_generator_with_line_spacing.py` and run it:
   ```
   python mri_report_generator_with_line_spacing.py
   ```

2. **Access the App**:
   Open your browser and navigate to `http://127.0.0.1:8050`.

3. **Upload an MRI Scan**:
   - Upload a pre-processed MRI scan (JPEG/PNG) with a YOLO bounding box.
   - Click "Generate Report" to view the generated report template with proper line spacing.

### Team
Developed by **ScanSquad**.
