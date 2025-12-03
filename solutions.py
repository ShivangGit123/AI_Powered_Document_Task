import os
import io
import json
from groq import Groq
from pypdf import PdfReader
from pydantic import BaseModel, Field, ConfigDict
from openpyxl import Workbook
import streamlit as st
from dotenv import load_dotenv

load_dotenv() 

LLM_MODEL = "llama-3.3-70b-versatile"

def init_client(api_key: str):
    """Initializes and returns the Groq client."""
    if not api_key:
        return None, False

    try:
        client = Groq(api_key=api_key)
        client.models.list() 
        return client, True
    except:
        return None, False



class ExtractedPair(BaseModel):
    """Represents a single row of data for the Excel output."""
    Key: str = Field(description="The determined key/label for the data point.")
    Value: str = Field(description="The exact original phrasing or fact from the PDF.")
    comment: str = Field(
        alias="Comment",
        description="Residual text from the source or supplementary context. MUST be '' if Key/Value is sufficient."
    )
    model_config = ConfigDict(populate_by_name=True)


class DocumentStructure(BaseModel):
    """The root model for the structured document output, enforcing the top-level key."""
    extracted_data: list[ExtractedPair] = Field(
        description="This array MUST contain ALL extracted Key:Value:Comment pairs."
    )



@st.cache_data
def read_pdf_text(uploaded_file: io.BytesIO) -> str:
    """Reads all text content from an uploaded PDF file."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"\n--- Page {i+1} ---\n"
            page_text = page.extract_text()
            text += page_text if page_text else "(No readable text on this page)"
        return text.strip()
    except Exception as e:
        # Remove cache on failure so the user can re-upload
        st.cache_data.clear()
        st.error(f"Error reading PDF: {e}")
        return ""


def generate_extraction_prompt(document_text: str) -> str:
    """
    Creates the optimized system prompt, using CoT and strict JSON instruction markers.
    """
   
    schema_template = DocumentStructure.model_json_schema(by_alias=True)
    
    return f"""
    You are an advanced AI Data Structuring and Extraction Engine. Your task is to transform the provided
    unstructured document text into a structured JSON format.

    ## SCHEMA DEFINITION (CRITICAL):
    Your final output MUST be a JSON object that STRICTLY conforms to the following schema structure. 
    The single top-level key MUST be "extracted_data" containing an array of objects.

    {json.dumps(schema_template, indent=2)}

    ## PROCESSING METHODOLOGY (MANDATORY CHAIN-OF-THOUGHT):
    You MUST process the document by following these steps for every unique sentence or clause:
    1.  **IDENTIFY SOURCE:** Select one complete, logical sentence or clause from the document text.
    2.  **EXTRACT CORE VALUE:** From that source, extract the single, most important factual metric or phrase. This is the **Value**.
    3.  **DETERMINE KEY:** Create the most appropriate, concise, and logical **Key** for the fact identified in Step 2.
    4.  **CAPTURE CONTEXT/RESIDUAL:** Place any remaining associated text from the *original source sentence* that was NOT used in the Key or Value into the **Comment**. If the Key and Value capture the entire logical idea, the comment MUST be **EMPTY ("")**.

    ## STRICT RULES FOR FIDELITY:
    1.  **100% Capture:** All content MUST be captured across the three columns (Key, Value, Comment). Nothing is summarized or omitted.
    2.  **Language Preservation (CRITICAL):**
        * The **Value** MUST **Retain the exact original wording, sentence structure, and phrasing from the PDF**.
        * **Avoid paraphrasing unless required to form a clean key:value pair**.
        * **Do not introduce new information** or fabricate details.
    3.  **Final Output:** You MUST output ONLY the JSON object. DO NOT include any introductory text, markdown, or commentary outside of the JSON block.

    ## DOCUMENT TEXT TO BE PROCESSED:
    ---
    {document_text}
    ---
    """


def extract_data_with_llm(client: Groq, document_text: str, progress_bar):
    system_prompt = generate_extraction_prompt(document_text)

    try:
        progress_bar.progress(60, text="60% - Sending data to Groq LLM...")

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "BEGIN EXTRACTION: Return the structured JSON immediately."}
            ],
            model=LLM_MODEL,
            response_format={"type": "json_object"},
            temperature=0.0
        )

        progress_bar.progress(80, text="80% - Validating JSON...")

        json_string = chat_completion.choices[0].message.content
        data_model = DocumentStructure.model_validate_json(json_string)

        progress_bar.progress(95, text="95% - Success!")

        return data_model.extracted_data

    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ LLM Error: {e}")
        st.caption("The LLM likely returned text/JSON that did not strictly contain the 'extracted_data' list. This is a model adherence issue.")
        return None


def create_excel_bytes(extracted_data, progress_bar):
    progress_bar.progress(98, text="98% - Generating Excel...")

    wb = Workbook()
    ws = wb.active
    ws.append(["Key", "Value", "Comment"])

    for item in extracted_data:
        row = item.model_dump(by_alias=True)
        ws.append([row["Key"], row["Value"], row["Comment"]])

    stream = io.BytesIO()
    wb.save(stream)

    progress_bar.progress(100, text="Done!")
    return stream.getvalue()


def main():
    st.set_page_config(page_title="AI Document Structuring Tool", layout="wide")
    st.title("📄 AI-Powered Document Structuring & Data Extraction")

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("### 🔑 Enter Your Groq API Key")

        api_key_input = st.text_input(
            label="Groq API Key",
            placeholder="Enter your GROQ_API_KEY here",
            type="password",
            key="API_KEY_INPUT", # Use a unique key for the widget
            value=st.session_state.get("API_KEY", os.environ.get("GROQ_API_KEY", "")),
            help="Your key is used only for this session's API calls."
        )
        
        st.session_state["API_KEY"] = api_key_input

        client, connected = init_client(st.session_state["API_KEY"])

        if connected:
            st.success(f"API Status: Connected to GroqCloud")
        else:
            st.error("API Status: Disconnected")
            st.info("Enter a valid API key in the field above to enable the LLM.")

        st.markdown("---")
        st.caption(f"LLM Model: **{LLM_MODEL}**")

    if not connected:
        st.warning("Please enter your API key in the sidebar to activate the application.")
        return

    st.markdown("## ➡️ Process Workflow")

    uploaded_file = st.file_uploader("**Step 1:** Upload 'Data Input.pdf'", type=["pdf"])

    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

        if uploaded_file.name != st.session_state.get('last_uploaded_file'):
             st.session_state['pdf_content'] = read_pdf_text(uploaded_file)
             st.session_state['last_uploaded_file'] = uploaded_file.name
        
        if 'pdf_content' not in st.session_state:
             st.session_state['pdf_content'] = read_pdf_text(uploaded_file)


        col1, col2 = st.columns(2)

        with col1:
            if st.button("2. 🔍 Preview Extracted Text", use_container_width=True):
                st.success("Text extracted successfully.")
                with st.expander("Show Raw PDF Text"):
                    st.code(st.session_state['pdf_content'][:1000] + " ...", language="text")

        with col2:
            run_extract = st.button("3. ⚡ Run LLM Extraction", type="primary", use_container_width=True)

        st.markdown("---")

        if run_extract:
            pdf_text = st.session_state['pdf_content']
            
            if not pdf_text:
                st.warning("Cannot proceed: PDF text content is empty.")
                return

            progress = st.progress(10, text="Starting LLM Processing...")

            data = extract_data_with_llm(client, pdf_text, progress)

            if data:
                st.subheader("✅ Step 4: Extraction Complete")

                excel_bytes = create_excel_bytes(data, progress)
                
                st.dataframe([d.model_dump(by_alias=True) for d in data], use_container_width=True, height=300)

                st.download_button(
                    "⬇️ Download Structured Output.xlsx",
                    excel_bytes,
                    "Structured_Output.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                progress.empty()
            
if __name__ == "__main__":
    main()

