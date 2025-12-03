import os
import io
import json
from groq import Groq
from pypdf import PdfReader
from pydantic import BaseModel, Field, ConfigDict
from openpyxl import Workbook
import streamlit as st
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

LLM_MODEL = "llama-3.3-70b-versatile" 

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        

class ExtractedPair(BaseModel):
    """
    Represents a single row of data for the Excel output. 
    The field is 'comment' internally but its alias in the JSON output will be 'Comment'.
    """
    
    Key: str = Field(description="The determined key/label for the piece of information, decided by the LLM.")
    Value: str = Field(description="The raw, EXACT value associated with the Key, preserving original wording.")
    comment: str = Field(
        alias="Comment",
        description="Additional text from the source providing context, or left empty ('') if Key/Value is sufficient."
    )
    
    model_config = ConfigDict(populate_by_name=True)

class DocumentStructure(BaseModel):
    """The root model for the structured document output."""
    extracted_data: list[ExtractedPair] = Field(description="A list containing all key:value pairs and their comments extracted from the entire document.")

# --- Core Functions ---

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
        st.error(f"An error occurred while reading the PDF: {e}")
        return ""

def generate_extraction_prompt(document_text: str) -> str:
    """
    Creates the detailed system prompt, utilizing Chain-of-Thought (CoT) prompting 
    for strict structural and linguistic fidelity.
    """
    
    return f"""
    You are an advanced AI Data Structuring and Extraction Engine. Your task is to transform the provided
    unstructured document text into a structured JSON format following the exact Pydantic schema provided.

    ## PROCESSING METHODOLOGY (MANDATORY CHAIN-OF-THOUGHT):
    You MUST process the document by following these steps for every unique sentence or clause:
    1.  **IDENTIFY SOURCE:** Select one complete, logical sentence or clause from the document text.
    2.  **EXTRACT CORE VALUE:** From that source, extract the single, most important factual metric or phrase. This is the **Value**.
    3.  **DETERMINE KEY:** Create the most appropriate, concise, and logical **Key** for the fact identified in Step 2.
    4.  **CAPTURE CONTEXT/RESIDUAL:** Place any remaining associated text from the *original source sentence* that was NOT used in the Key or Value into the **Comment**. If the Key and Value capture the entire logical idea, the comment MUST be **EMPTY ("")**.

    ## STRICT RULES FOR FIDELITY:
    1.  **100% Data Capture:** All content MUST be captured across the three columns (Key, Value, Comment). Nothing is summarized or omitted.
    2.  **Language Preservation (CRITICAL):**
        * The **Value** MUST **Retain the exact original wording, sentence structure, and phrasing from the PDF**.
        * **Avoid paraphrasing unless required to form a clean key:value pair**.
        * **Do not introduce new information** or fabricate details.
    3.  **Schema Enforcement:** The final JSON object MUST have a **single top-level key** named **'extracted_data'**. DO NOT categorize the data into custom keys.

    ## EXAMPLE OUTPUT FORMAT:
    // Note the use of "Comment" as the field name, reflecting the alias.
    {{
        "extracted_data": [
            {{
                "Key": "Assignment Title",
                "Value": "AI-Powered Document Structuring & Data Extraction Task",
                "Comment": "" 
            }},
            {{
                "Key": "Undergraduate Graduation Context",
                "Value": "Graduating with honors",
                "Comment": "ranking 15th among 120 students in his class." 
            }}
        ]
    }}

    ## DOCUMENT TEXT TO BE PROCESSED (Chunked Input):
    ---
    {document_text}
    ---
    """

def extract_data_with_llm(document_text: str, progress_bar) -> list[ExtractedPair] | None:
    """Calls the Groq API to perform the structured data extraction."""
    
    system_prompt = generate_extraction_prompt(document_text)
    
    try:
        progress_bar.progress(60, text="60% - Sending data to Groq LLM for extraction...")
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Process the provided document text and return the structured JSON output adhering strictly to the Pydantic schema."}
            ],
            model=LLM_MODEL,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        progress_bar.progress(80, text="80% - Validating and parsing LLM output...")
        
        json_string = chat_completion.choices[0].message.content
        data_model = DocumentStructure.model_validate_json(json_string)
        
        progress_bar.progress(95, text="95% - Data validated successfully.")
        return data_model.extracted_data
        
    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ An error occurred during LLM API call or JSON parsing. Details: {e}")
        st.caption("The model failed to adhere to the strict JSON schema (likely missing the 'extracted_data' key or violating the list structure).")
        return None

def create_excel_bytes(extracted_data: list[ExtractedPair], progress_bar) -> bytes:
    """Creates an Excel file in memory (BytesIO) and returns the bytes."""
    progress_bar.progress(98, text="98% - Generating Excel file in memory...")
    
    wb = Workbook()
    ws = wb.active
    
    headers = ["Key", "Value", "Comment"] 
    ws.append(headers)
    
    # Write data rows
    for item in extracted_data:
        row_dict = item.model_dump(by_alias=True) 
        row = [row_dict['Key'], row_dict['Value'], row_dict['Comment']]
        ws.append(row)
        
    excel_stream = io.BytesIO()
    wb.save(excel_stream)
    excel_bytes = excel_stream.getvalue()
    
    progress_bar.progress(100, text="100% - File ready for download!")
    return excel_bytes



def main():
    st.set_page_config(page_title="AI Document Structuring Tool", layout="wide")
    st.title("📄 AI-Powered Document Structuring & Data Extraction")
    st.markdown("A solution to convert unstructured PDF text into a structured Excel format, emphasizing **100% data fidelity** and **original language preservation**.")

    
    with st.sidebar:
        st.header("⚙️ Configuration")
        if client:
            st.success("API Status: Connected to GroqCloud")
            st.info(f"LLM Model: **{LLM_MODEL}**")
            st.info("Strategy: Explicit Chain-of-Thought (CoT) Prompting")
        else:
            st.error("API Status: Disconnected (Key Missing)")
            st.warning("Please set your `GROQ_API_KEY` environment variable in the `.env` file.")
        st.markdown("---")
        st.caption("Using Llama 3.3 for high-fidelity extraction after Mixtral was decommissioned.")

    if client is None:
        st.warning("""
            **Application Disabled:** Cannot connect to the LLM. 
            Please set your Groq API Key to proceed.
        """)
        return

    # --- Main Application Steps ---
    st.markdown("## ➡️ Process Workflow")
    
    uploaded_file = st.file_uploader(
        "**Step 1:** Upload your 'Data Input.pdf' here:", 
        type=["pdf"], 
        key="file_uploader",
        help="The document must be text-readable (not a scanned image)."
    )
    
    if uploaded_file is not None:
        st.success(f"File uploaded: **{uploaded_file.name}**")
        
        # --- Step 2 & 3: Read and Run Buttons ---
        col1, col2 = st.columns(2)
        
        # Ensure content is extracted before processing
        if 'pdf_content' not in st.session_state or st.session_state.get('last_uploaded_file') != uploaded_file.name:
            st.session_state['pdf_content'] = read_pdf_text(uploaded_file)
            st.session_state['last_uploaded_file'] = uploaded_file.name

        with col1:
            if st.button("2. 🔍 Preview Extracted Text", use_container_width=True, key="btn_preview"):
                st.success("Text extraction complete.")
                with st.expander("Review Extracted Text"):
                    st.code(st.session_state['pdf_content'][:1000] + "\n...", language="text")

        with col2:
            process_button = st.button("3. ⚡ Run Data Structuring (LLM)", type="primary", use_container_width=True, key="btn_process")
            
        st.markdown("---")

        # --- Step 4: Processing Logic and Results ---
        
        if process_button:
            pdf_content = st.session_state['pdf_content']
            
            if not pdf_content:
                st.warning("Text extraction failed. Cannot proceed.")
                return

            # Initialize progress bar
            progress_bar = st.progress(10, text="10% - Starting LLM processing...")
            
            # Extract data using LLM
            structured_data = extract_data_with_llm(pdf_content, progress_bar)

            if structured_data:
                st.subheader("✅ Extraction Complete: Results")
                
                # Create downloadable Excel
                excel_bytes = create_excel_bytes(structured_data, progress_bar)
                
                st.success(f"**{len(structured_data)}** Key:Value pairs captured and structured.")

                st.dataframe([d.model_dump(by_alias=True) for d in structured_data], use_container_width=True, height=300)

                st.download_button(
                    label="⬇️ Download Expected Output.xlsx",
                    data=excel_bytes,
                    file_name="Expected_Output_Structured_Data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Final output adhering to all fidelity and structuring requirements.",
                    key="download_button"
                )
                progress_bar.empty()
            
if __name__ == "__main__":
    main()
