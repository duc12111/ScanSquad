import io
import base64
import io

import numpy as np
import pydicom
from PIL import Image
from dash import Dash, html, dcc, Output, Input, State, no_update

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

        html.Div(id='output-images',
                 style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
                        "gap": "1rem"})
    ]
)


def dicom_to_png_bytes(dicom_bytes):
    try:
        dcm = pydicom.dcmread(io.BytesIO(dicom_bytes))
        pixel_array = dcm.pixel_array

        # Normalize pixel values
        pixel_array = pixel_array.astype(float)
        pixel_array -= np.min(pixel_array)
        pixel_array /= np.max(pixel_array)
        pixel_array *= 255.0

        image = Image.fromarray(pixel_array.astype(np.uint8))

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
                    style={"width": "100%", "borderRadius": "10px", "boxShadow": "0 4px 8px rgba(0,0,0,0.1)"}
                )
                images.append(img_element)
        return images
    return no_update


if __name__ == '__main__':
    app.run(debug=True)
