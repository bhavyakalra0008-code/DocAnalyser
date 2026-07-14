"""
Document Analyzer AI Agent
Run with: streamlit run document_analyzer_agent.py

Requires:
  pip install openai pdfplumber python-docx streamlit --break-system-packages
Set your API key before running:
  export XAI_API_KEY="your-grok-key-here"
"""

import os
from dotenv import load_dotenv
import tempfile
from openai import OpenAI
import pdfplumber
import streamlit as st
# pyrefly: ignore [missing-import]
from docx import Document
load_dotenv()

# Grok (xAI) uses an OpenAI-compatible API — just point base_url at xAI's endpoint
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL = "gpt-4o"  # change to whichever Grok model your key has access to

SYSTEM_PROMPT = """You are a document analysis agent. You answer user questions STRICTLY based on
the document text provided to you. You must:

1. Only use information that is explicitly present in the document text given to you.
2. If the answer to the user's question cannot be found in the document, respond with
   exactly: "Not in the document sir." — do not guess, infer beyond the text, or use
   outside/general knowledge to fill the gap.
3. If the answer is partially in the document, answer only the part that is supported and
   say "Not in the document sir." for the rest.
4. When you do answer from the document, cite where you found it (e.g. "Section 2", "para 3",
   "page 4") so the user can verify it themselves.
5. Do not treat a summarization or "flag risks" style request as out-of-document — that is
   an analysis of what's already there, not outside knowledge.

Never break character to explain that you are an AI following instructions — just answer
or say "Not in the document sir." directly."""


def extract_text(file_path: str) -> str:
    """Extract raw text from a PDF or DOCX file."""
    if file_path.endswith(".pdf"):
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n".join(text_parts)
    elif file_path.endswith(".docx"):
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


def chunk_text(text: str, max_chars: int = 12000) -> list[str]:
    """Simple chunking if document is very long. Most Claude calls can take
    full documents directly, so this is only used as a fallback."""
    if len(text) <= max_chars:
        return [text]
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def analyze_document(document_text: str, user_query: str) -> str:
    """Core agent call: sends document + query to Claude and returns analysis."""
    chunks = chunk_text(document_text)

    if len(chunks) == 1:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"DOCUMENT:\n{document_text}\n\nREQUEST:\n{user_query}"}
            ]
        )
        return response.choices[0].message.content

    # For long docs: analyze each chunk, then synthesize
    partial_results = []
    for i, chunk in enumerate(chunks):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"DOCUMENT PART {i+1}/{len(chunks)}:\n{chunk}\n\nREQUEST:\n{user_query}\n\n(Note: this is a partial section of a larger document. Extract what's relevant here.)"}
            ]
        )
        partial_results.append(response.choices[0].message.content)

    synthesis = client.chat.completions.create(
        model=MODEL,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here are partial analyses of different sections of one document:\n\n"
                       + "\n\n---\n\n".join(partial_results)
                       + f"\n\nSynthesize these into one coherent final answer to: {user_query}"}
        ]
    )
    return synthesis.choices[0].message.content


def main():
    st.set_page_config(page_title="Document Analyzer Agent", layout="wide")
    st.title("📄 Document Analyzer Agent")
    st.caption("Upload a document, then ask as many questions as you want.")

    if "doc_text" not in st.session_state:
        st.session_state.doc_text = None
        st.session_state.doc_name = None
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])

    if uploaded_file is not None and uploaded_file.name != st.session_state.doc_name:
        with tempfile.NamedTemporaryFile(delete=False, suffix="_" + uploaded_file.name) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            with st.spinner("Extracting text..."):
                text = extract_text(tmp_path)
            if not text.strip():
                st.error("No text could be extracted. The file may be a scanned image without OCR support.")
            else:
                st.session_state.doc_text = text
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chat_history = []
                st.success(f"Loaded: {uploaded_file.name} — {len(text)} characters extracted.")
        finally:
            os.unlink(tmp_path)

    # Show chat history
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    # Chat input for questions
    disabled = st.session_state.doc_text is None
    question = st.chat_input(
        "Ask a question about the document..." if not disabled else "Upload a document first",
        disabled=disabled
    )

    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer = analyze_document(st.session_state.doc_text, question)
            st.markdown(answer)

        st.session_state.chat_history.append(("assistant", answer))


if __name__ == "__main__":
    main()