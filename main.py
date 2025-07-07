import streamlit as st
from typing import TypedDict
import pdfplumber
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from langgraph.graph import StateGraph, END
from google.oauth2.service_account import Credentials
from google.auth import default
from langchain_community.llms import HuggingFaceHub
# from langchain_community.llms import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import pipeline
from huggingface_hub import InferenceClient

import time
import json

HUGGINGFACEHUB_API_TOKEN = st.secrets["huggingface"]["api_token"]

client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=HUGGINGFACEHUB_API_TOKEN)

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", use_auth_token=HUGGINGFACEHUB_API_TOKEN)
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", use_auth_token=HUGGINGFACEHUB_API_TOKEN, torch_dtype="auto", device_map="auto")

# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# llm = HuggingFacePipeline(pipeline=pipe)



# llm = HuggingFaceHub(
#     repo_id = "google/flan-t5-xl",
#     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
#     model_kwargs={
#         "temperature": 0.5,
#         "max_new_tokens": 512
#     }

# )

# Set up Google Cloud credentials
# raw = st.secrets["google"]["credentials"]
# service_account_info = json.loads(raw) 
# credentials = Credentials.from_service_account_info(service_account_info, scopes=['https://www.googleapis.com/auth/cloud-platform'])

# llm = VertexAI(
#     project = "machine-translation-001",
#     location = "us-central1",
#     model = "gemini-2.5-pro-preview-05-06",
#     credentials=credentials
# )


class ContractRiskState(TypedDict):
    contract_path: st.runtime.uploaded_file_manager.UploadedFile
    rules_path: st.runtime.uploaded_file_manager.UploadedFile
    contract_text: str
    rules_text: str
    risks_detected: str

def extract_text_node(state: ContractRiskState) -> ContractRiskState:
    with pdfplumber.open(state["contract_path"]) as contract_pdf:
        text_contract = "".join(page.extract_text() or "" for page in contract_pdf.pages)

    with pdfplumber.open(state["rules_path"]) as rules_pdf:
        text_rules = "".join(page.extract_text() or "" for page in rules_pdf.pages)

    return {
        **state,
        "contract_text": text_contract,
        "rules_text": text_rules
    }


# Set up Google Cloud credentials
# Creating prompt and inference the output

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/pyaephyopaing/Desktop/ET.Verdict/MulitAgent_LangGraph/botexpert-459607-26d1eb471087.json"




prompt = PromptTemplate(
    input_variables=["contract_text", "rules_text"],
    template = """
You are an expert in contract analysis, and your task is to identify potential **risks** in contract clauses.

You have been provided with:
1. A contract document to review.
2. The company's internal rules and regulations that must be adhered to.

---
Your job is to:
- Carefully analyze the contract clauses.
- Cross-check each clause against the company's rules and regulations.
- Detect and extract **any sentences or clauses** that may present **legal, regulatory, or operational risk**, especially where they **violate or contradict the rules**.
- For each risky clause you find:
    - Identify the **clause type** (e.g., Confidentiality, Termination, Payment, etc.).
    - Include the exact **sentence or clause** from the contract.


Pay close attention to the following clause types:
- Confidentiality Clause
- Indemnification Clause
- Force Majeure Clause
- Dispute Resolution Clause
- Arbitration Clause
- Termination Clause
- Jurisdiction Clause
- Privacy Clause
- Warranty and Disclaimer Clause
- Damages Clause
- Payment Clause
- Data Protection and Privacy Clause
- Conflicts of Interest Clause
- Choice of Law Clause
- Change Control Clause
- Penalty Clause
- Non-Compete Clause
- Subcontracting Clause
- Severability Clause
- Statute of Limitations Clause

---
Contract:
{contract_text}

---
Company Rules and Regulations:
{rules_text}

---
Please return the risks in **exactly** the following format:

- [Clause Type: <ClauseType>]
    Risk sentence {{1}}: "<Exact sentence or clause>"

- [Clause Type: <ClauseType>]
    Risk sentence {{2}}: "<Exact sentence or clause>"

Each result should:
- Start with the clause type in square brackets on its own line.
- Follow with the risky sentence on the next line (indented or quoted).
- Use this format consistently without skipping lines or adding unnecessary explanation.

"""
)



def detect_risks_node(state: ContractRiskState) -> ContractRiskState:
    formatted_prompt = prompt.format_prompt(
        contract_text=state["contract_text"],
        rules_text = state["rules_text"]
        ).to_string()
    # response = llm.invoke(formatted_prompt)

    response = client.text_generation(
        prompt=formatted_prompt,
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.9,
        stop_sequences=["\n\n"]
    )

    return {
        **state,
        "risks_detected": response
    }


# Create state graph
workflow = StateGraph(ContractRiskState)

workflow.add_node("extract_text", extract_text_node)
workflow.add_node("detect_risks", detect_risks_node)

workflow.set_entry_point("extract_text")
workflow.add_edge("extract_text", "detect_risks")
workflow.set_finish_point("detect_risks")

app = workflow.compile()


def main():
    st.title("Contract Risk Detection")

    contract_path = st.file_uploader("Upload Contract File", type="pdf")
    rules_path = st.file_uploader("Upload Rules & Regulations File", type="pdf")

    if contract_path and rules_path:
        if st.button("Start Detect Risks"):
            # Show progress bar
            progress_text = "Analyzing contract... Please wait."
            my_bar = st.progress(0, text=progress_text)

            # Simulate progress (0 to 90%) while processing
            for percent_complete in range(90):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)

            # Call the real LLM function
            result = app.invoke({
                "contract_path": contract_path,
                "rules_path": rules_path
            })

            # Complete progress bar
            my_bar.progress(100, text="Done! Displaying risks...")
            time.sleep(0.5)
            my_bar.empty()

            # Show result
            st.subheader("Detected Risks:")
            st.markdown(result["risks_detected"])

if __name__ == "__main__":
    main()
