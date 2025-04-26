from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import base64
import os
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)  # load api key

# Function for decoding image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit for uploads

app.layout = dbc.Container(
    [
        dcc.Markdown("## MRI Aneurysm Report Generator\n"
                     "###### Upload a pre-processed MRI scan (JPEG/PNG) with a YOLO bounding box indicating positive/negative cases for aneurysms."),
        dcc.Upload(
            id='upload-image',
            children=html.Div(['Drag and Drop or Select File (MRI Scan with Bounding Box)']),
            style={
                'width': '50%',
                'height': 'auto',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False  # Single file upload
        ),
        dcc.Store(id='session-memory', data=[]),  # Store for session memory
        html.Div(id='image-display'),
        dbc.Row(
            [
                dbc.Col(dbc.Button(id='btn', children='Generate Report', className='my-2'), width=2)
            ],
        ),
        dbc.Row(
            [
                dbc.Col(dbc.Spinner(dcc.Markdown(id='content', children=''), fullscreen=False), width=6)
            ],
        ),
    ]
)

# Callback to display the uploaded image (unchanged)
@app.callback(
    Output('image-display', 'children'),
    Input('upload-image', 'contents')
)
def display_image(contents):
    if not contents:
        return 'No image uploaded.'

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Save the image temporarily to display
    temp_path = "temp_image.jpg"
    with open(temp_path, 'wb') as f:
        f.write(decoded)

    return html.Div([
        html.P("Uploaded MRI Scan with Bounding Box:"),
        html.Img(src='data:image/png;base64,{}'.format(content_string),
                 style={'width': '50%', 'height': 'auto'})
    ])

# Callback to trigger API call when Generate Report button is clicked
@callback(
    Output('content', 'children'),
    Output('session-memory', 'data'),
    Input('btn', 'n_clicks'),
    State('upload-image', 'contents'),
    State('session-memory', 'data'),
    prevent_initial_call=True
)
def generate_mri_report(n_clicks, contents, session_memory):
    if n_clicks is None or not contents:
        return 'Please upload an image first.', session_memory

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Save the image temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, 'wb') as f:
        f.write(decoded)

    base64_image = encode_image(temp_path)

    # Clean up
    os.remove(temp_path)

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

    prompt_prefix = (
        "You are an assistant that helps generate MRI report templates based on visual observations of pre-processed MRI scans.\n"
        "The provided MRI scan includes a bounding box from a YOLO model highlighting an area of interest, which may indicate a potential aneurysm or other abnormality.\n"
        "First, describe the visual features of the MRI scan and the bounding box (e.g., location, size, shape, contrast, intensity patterns). Include your confidence level in these observations (e.g., high, moderate, low confidence).\n"
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
            "text": prompt_prefix + "Analyze the following MRI scan and generate a report template based on its visual features:"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "auto"
            }
        }
    ]

    # Call the Open AI model
    try:
        chat = ChatOpenAI(
            model="gpt-4o",  # Vision-capable model
            max_tokens=1024,
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

if __name__ == '__main__':
    app.run_server(debug=True)