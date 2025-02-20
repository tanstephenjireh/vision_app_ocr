import streamlit as st
import ollama
from openai import OpenAI
from PIL import Image
import os
from dotenv import load_dotenv
import base64

load_dotenv()

system_prompt = """Analyze the text in the provided image. Extract all readable content
                and present it in a structured Markdown format that is clear, concise, 
                and well-organized. Ensure proper formatting (e.g., headings, lists, or
                code blocks) as necessary to represent the content effectively."""

# Function to encode the image
def encode_image(image_path):
    """Encode uploaded file bytes to base64"""
    return base64.b64encode(image_path.getvalue()).decode("utf-8")


def ocr_openai(image):
    if st.button("Extract Text using OpenAI 🔍", type="primary"):
        with st.spinner("Processing image..."):
            base64_image = encode_image(image)
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": system_prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                )
                st.session_state['ocr_result'] = response.choices[0].message.content
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

def ocr_llama(image):
    if st.button("Extract Text using LLaMa 🔍", type="primary"):
        with st.spinner("Processing image..."):
            try:
                response = ollama.chat(
                    model='llama3.2-vision',
                    messages=[{
                        'role': 'user',
                        'content': system_prompt,
                        'images': [image]
                    }]
                )
                st.session_state['ocr_result'] = response.message.content
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="Llama OCR",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description in main area
st.title("OpenAI or 🦙 Llama OCR")

# Add clear button to top right
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear 🗑️"):
        if 'ocr_result' in st.session_state:
            del st.session_state['ocr_result']
        st.rerun()

st.markdown('<p style="margin-top: -20px;">Extract structured text from images using OpenAI gpt-4o model or Llama 3.2 Vision!</p>', unsafe_allow_html=True)
st.markdown("---")

# Move upload controls to sidebar
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        # st.write(uploaded_file)
        # st.write(uploaded_file.getvalue())
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        # Choose model logic
        model_type = st.selectbox("Select Model Type", ["OpenAI", "LLaMA"]) 
        if model_type == "OpenAI":
            ocr_openai(uploaded_file)
        
        elif model_type == "LLaMA":
            ocr_llama(uploaded_file.getvalue())

# Main content area for results
if 'ocr_result' in st.session_state:
    st.markdown(st.session_state['ocr_result'])
else:
    st.info("Upload an image and click 'Extract Text' to see the results here.")

# Footer
st.markdown("---")
st.markdown("Made using Llama Vision Model2 & OpenAI vision model")
# st.markdown("Made with ❤️ using Llama Vision Model2 | [Report an Issue](link)")