from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import io
import base64
import pdf2image
import google.generativeai as genai
from deep_translator import GoogleTranslator

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def google_translate(text, source_lang, target_lang):
    """
    Translates text using Google Translate API.

    Args:
        text (str): The text to translate.
        source_lang (str): The source language code.
        target_lang (str): The target language code.

    Returns:
        str: The translated text.
    """
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    translated = translator.translate(text)
    return translated

def input_pdf_setup(uploaded_file):
    """
    Converts uploaded PDF to a list of dictionaries containing image data.

    Args:
        uploaded_file (streamlit.UploadedFile): The uploaded PDF file.

    Returns:
        list: A list of dictionaries containing image data ({'mime_type': 'image/jpeg', 'data': base64-encoded image data})
              or raises FileNotFoundError if no file is uploaded.
    """
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        pdf_parts = []
        for image in images:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            pdf_parts.append({
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            })
        return pdf_parts
    else:
        raise FileNotFoundError("No file uploaded")

def process_gemini_response(response):
    """
    Processes the response from the Gemini API, handling potential errors or empty responses.

    Args:
        response (genai.TextResponse): The response object from the Gemini API.

    Returns:
        str: The extracted text from the response (if successful), or an error message.
    """
    if not response.candidates:
        if response.prompt_feedback:
            for rating in response.prompt_feedback.safety_ratings:
                if rating.probability != "NEGLIGIBLE":
                    return f"Content was blocked due to {rating.category} with {rating.probability} probability."
        return "The response was blocked or empty."

    candidate = response.candidates[0]
    if candidate.content is None or candidate.content.parts is None:
        return "The response did not contain any content."

    text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
    if not text_parts:
        return "The response did not contain any text parts."

    return " ".join(text_parts)

def explain_and_summarize(text):
    """
    Explains and summarizes the given text using the Gemini API.

    Args:
        text (str): The text to explain and summarize.

    Returns:
        str: The explanation and summary from Gemini (if successful), or an error message.
    """
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"""
    Please explain and summarize the following text in simple, easy-to-understand language. 
    Keep the summary concise but include all key points:

    {text}
    """
    try:
        response = model.generate_content(prompt)
        return process_gemini_response(response)
    except Exception as e:
        return f"An error occurred while generating the explanation: {str(e)}"

st.title("Document Translation and Explanation Service")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
source_lang = st.selectbox("Source Language", ["en", "fr", "es", "de", "it"], key="source_lang")
target_lang = st.selectbox("Target Language", ["en", "fr", "es", "de", "it"], key="target_lang")

if uploaded_file is not None:
    if st.button("Translate", key="translate_button"):
        try:
            with st.spinner("Processing PDF..."):
                pdf_parts = input_pdf_setup(uploaded_file)
            
            with st.spinner("Extracting text..."):
                model = genai.GenerativeModel('gemini-pro-vision')
                extracted_text = ""
                for part in pdf_parts:
                    response = model.generate_content([part, "Extract all the text from this image"])
                    extracted_text += process_gemini_response(response) + "\n\n"
            
            with st.spinner("Translating..."):
                translated_text_str = google_translate(extracted_text, source_lang, target_lang)
            
            st.success("Translation complete!")
            
            st.subheader("Original Text")
            st.text_area("Original Text", extracted_text, height=200, key="original_text")
            
            st.subheader("Translated Text")
            st.text_area("Translated Text", translated_text_str, height=200, key="translated_text")
            
            # Add Explain and Summarize button
            if st.button("Explain and Summarize", key="explain_button"):
                with st.spinner("Generating explanation and summary..."):
                    explanation = explain_and_summarize(translated_text_str)
                st.subheader("Explanation and Summary")
                st.write(explanation)
                
                # Only show download button if explanation is not an error message
                if not explanation.startswith("An error occurred") and not explanation.startswith("Content was blocked"):
                    st.download_button(
                        label="Download explanation",
                        data=explanation,
                        file_name="explanation_summary.txt",
                        mime="text/plain",
                        key="download_explanation"
                    )
                else:
                    st.warning("Explanation could not be generated or downloaded due to the error above.")
            
            # Download button for translated text
            st.download_button(
                label="Download translated text",
                data=translated_text_str,
                file_name="translated_document.txt",
                mime="text/plain",
                key="download_translated"
            )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please check your API keys and try again.")
else:
    st.warning("Please upload a PDF file")