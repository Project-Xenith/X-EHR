from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import io
import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from huggingface_hub import login
from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import display, Markdown
load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_bucket = os.getenv("BUCKET_NAME")
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")
login(token=hugging_face_api_key)
supabase = create_client(supabase_url, supabase_key)
FILE_NAME_1 = "combined_output_brotli_2.parquet"
FILE_NAME_2 = "combined_output_brotli_1.parquet"
df1 = pd.read_parquet(io.BytesIO(supabase.storage.from_(supabase_bucket).download(FILE_NAME_1)))
df2 = pd.read_parquet(io.BytesIO(supabase.storage.from_(supabase_bucket).download(FILE_NAME_2)))
df = pd.concat([df1, df2], ignore_index=True)
relevant_cols = [
'patient_id',
'med_date_service',
'diag_diagnosis_code',
'proc_procedure_code',
'total_submitted_cost',
'total_paid_cost',
'enroll_benefit_type']
df_clean = df[relevant_cols].copy()

df_clean.dropna(subset=['patient_id', 'med_date_service'], inplace=True)

# Step 3: Correct datatypes
df_clean['med_date_service'] = pd.to_datetime(df_clean['med_date_service'], errors='coerce')
df_clean['total_submitted_cost'] = pd.to_numeric(df_clean['total_submitted_cost'], errors='coerce')
df_clean['total_paid_cost'] = pd.to_numeric(df_clean['total_paid_cost'], errors='coerce')
df_clean.dropna(subset=['med_date_service', 'total_submitted_cost', 'total_paid_cost'], inplace=True)

# Load Mistral model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True  # This is what keeps it lightweight
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


def load_patient_data(patient_id):

    # Step 5: Optional — group by patient + date to consolidate multiple rows
    df_grouped = (
        df_clean
        .groupby(['patient_id', 'med_date_service'])
        .agg({
            'diag_diagnosis_code': lambda x: ', '.join(set(x.dropna().astype(str))),
            'proc_procedure_code': lambda x: ', '.join(set(x.dropna().astype(str))),
            'total_submitted_cost': 'sum',
            'total_paid_cost': 'sum',
            'enroll_benefit_type': lambda x: ', '.join(sorted(set(x.dropna())))
        })
        .reset_index()
    )
    return df_grouped[df_grouped['patient_id'] == patient_id]

def format_prompt(content: str) -> str:
    return f"<s>[INST] {content} [/INST]"

def healthcare_user_prompt(text: str) -> str:
    return (
        f"Summarize the following healthcare records in clear, concise language suitable for patients:\n{text}\n"
        "Avoid repetition and focus on key medical events."
    )
def chunk_text(text, max_tokens=768):
    tokens = tokenizer.encode(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    records = text.split(". ")
    for record in records:
        record_tokens = tokenizer.encode(record + ". ")
        if current_token_count + len(record_tokens) > max_tokens:
            if current_chunk:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = []
                current_token_count = 0
        current_chunk.append(record)
        current_token_count += len(record_tokens)
    if current_chunk:
        chunks.append(". ".join(current_chunk) + ".")
    return chunks
def combine_summaries(summaries):
    seen = set()
    combined = []
    for summary in summaries:
        summary = summary.replace("Healthcare summary for the patient.", "").strip()
        if summary and summary not in seen:
            seen.add(summary)
            combined.append(summary)
    return "\n".join(combined) if combined else "No meaningful summary generated."
def filter_ui(df):
    min_date = df_grouped['med_date_service'].min().date()
    max_date = df_grouped['med_date_service'].max().date()

    start_date = widgets.DatePicker(description="Start Date", value=min_date)
    end_date = widgets.DatePicker(description="End Date", value=max_date)

    diag_input = widgets.Text(description="Diagnosis:", placeholder="e.g., D123")
    proc_input = widgets.Text(description="Procedure:", placeholder="e.g., P456")
    enroll_type= widgets.Dropdown(
    options=['PHARMACY', 'MEDICAL'],
    description='Enroll Type:',
    style={'description_width': 'initial'}  # Makes label fully visible
)

    
    display(start_date, end_date, diag_input, proc_input,enroll_type)

    def apply_filters():
        dff = df_grouped.copy()
        if start_date.value:
            dff = dff[dff['med_date_service'] >= pd.to_datetime(start_date.value)]
        if end_date.value:
            dff = dff[dff['med_date_service'] <= pd.to_datetime(end_date.value)]
        if diag_input.value:
            dff = dff[dff['diag_diagnosis_code'].str.contains(diag_input.value, na=False)]
        if proc_input.value:
            dff = dff[dff['proc_procedure_code'].str.contains(proc_input.value, na=False)]
        if enroll_type.value:
            dff = dff[dff['enroll_benefit_type'].str.contains(enroll_type.value, na=False)]
            
        return dff
    
    return apply_filters
def generate_healthcare_summary(filtered_df, patient_id):
    if filtered_df.empty:
        return "### 🏥 No healthcare records found."

    text = ""
    for _, row in filtered_df.iterrows():
        text += f"On {row['med_date_service'].date()}, diagnosis code {row['diag_diagnosis_code']} and procedure code {row['proc_procedure_code']} were recorded. "

    chunks = chunk_text(text)
    summaries = []
    for chunk in tqdm(chunks, desc="Summarizing"):
        prompt = format_prompt(healthcare_user_prompt(chunk))
        response = pipe(prompt, max_new_tokens=200, temperature=0.7)[0]['generated_text']
        summaries.append(response.split('[/INST]')[-1].strip())

    combined = combine_summaries(summaries)
    return f"""### 🏥 Healthcare Summary for Patient ID `{patient_id}`

{combined}
"""
def generate_cost_summary(filtered_df, patient_id):
    if filtered_df.empty:
        return "### 💵 No cost records found."

    total_cost = filtered_df['total_submitted_cost'].sum()
    total_paid = filtered_df['total_paid_cost'].sum()
    
    breakdown = "\n".join([
        f"- {row['med_date_service'].date()}: Procedure(s): {row['proc_procedure_code']} | Submitted: ${row['total_submitted_cost']:.2f}, Paid: ${row['total_paid_cost']:.2f}"
        for _, row in filtered_df.iterrows()
    ])


    return f"""### 💵 Cost Summary for Patient ID `{patient_id}`

**Total Submitted:** ${total_cost:.2f}  
**Total Paid:** ${total_paid:.2f}  

**Breakdown:**
{breakdown}
"""
patient_id = input("Enter Patient ID: ").strip()
df_grouped = load_patient_data(patient_id)

if df_grouped.empty:
    print("No records found for this patient.")
else:
    print("Apply filters to generate summaries")
    get_filtered_df = filter_ui(df_grouped)

    run_button = widgets.Button(description="Generate Summaries")
    output = widgets.Output()

    def on_button_clicked(b):
        with output:
            output.clear_output()
            filtered = get_filtered_df()
            display(Markdown(generate_healthcare_summary(filtered, patient_id)))
            display(Markdown(generate_cost_summary(filtered, patient_id)))

    run_button.on_click(on_button_clicked)
    display(run_button, output)




