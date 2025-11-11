from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from config.settings import get_settings
from typing import List
from tqdm import tqdm

settings = get_settings()

tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    settings.MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
    # load_in_4bit=True
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

def format_prompt(content: str) -> str:
    return f"<s>[INST] {content} [/INST]"

def healthcare_prompt(text: str) -> str:
    return (
        f"Summarize the following healthcare records in clear, patient-friendly language. "
        f"Avoid jargon and repetition:\n{text}\nFocus on key events, diagnoses, and costs."
    )

def chunk_text(text: str, max_tokens: int = 768) -> List[str]:
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    words = text.split()
    chunks = []
    current = []
    count = 0
    for word in words:
        word_tokens = len(tokenizer.encode(" " + word))
        if count + word_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = [word]
            count = word_tokens
        else:
            current.append(word)
            count += word_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks

def generate_medical_summary(records: List[dict]) -> str:
    if not records:
        return "No medical records found."
    
    text = " ".join([
        f"On {r['date']}, diagnosis: {r['diag']}, procedure: {r['proc']}."
        for r in records
    ])
    chunks = chunk_text(text)
    summaries = []
    for chunk in tqdm(chunks, desc="Summarizing", leave=False):
        prompt = format_prompt(healthcare_prompt(chunk))
        result = pipe(prompt, temperature=0.7, do_sample=True)[0]['generated_text']
        summaries.append(result.split('[/INST]')[-1].strip())
    
    seen = set()
    combined = []
    for s in summaries:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            combined.append(s)
    return "\n".join(combined) or "No summary generated."