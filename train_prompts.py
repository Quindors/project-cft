#!/usr/bin/env python3
# train_prompts.py
# pip install gspread google-auth openai python-dotenv

import os, time, base64, tempfile, gspread, openai
from datetime import date, timedelta
from typing import List, Dict
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

load_dotenv(override=True)

# --- Configurations ---
SHEET_URL_OR_KEY = "https://docs.google.com/spreadsheets/d/1GU5H7sB0u2ximxylH-E-3qx0DcT3dNpqiM5lztuVNdg/edit"
WORKSHEET_PREFIX = "Focus Logs"
MODEL_OPTIMIZER = "gpt-4o"  # Use a strong model for optimization
PROMPT_MAIN_FILE = "prompt-main.txt"
PROMPT_CRITIC_FILE = "prompt-critic.txt"

# --- Auth Reuse (from monitor-sheets.py) ---
b64 = os.getenv("GCP_SERVICE_ACCOUNT_B64")
if not b64:
    raise RuntimeError("GCP_SERVICE_ACCOUNT_B64 not set.")

data = base64.b64decode(b64)
tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
tmp.write(data)
tmp.flush()
tmp.close()
SERVICE_ACCOUNT_JSON = tmp.name

openai.api_key = os.getenv("OPENAI_API_KEY")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

def get_client():
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=SCOPES)
    gc = gspread.authorize(creds)
    sh = gc.open_by_url(SHEET_URL_OR_KEY) if SHEET_URL_OR_KEY.startswith("http") else gc.open_by_key(SHEET_URL_OR_KEY)
    return gc, sh

def fetch_mistakes(sh, days_back=3) -> List[Dict]:
    mistakes = []
    # Check last N days
    for i in range(days_back):
        d = date.today() - timedelta(days=i)
        title = f"{WORKSHEET_PREFIX} - {d:%Y-%m-%d}"
        try:
            ws = sh.worksheet(title)
        except gspread.exceptions.WorksheetNotFound:
            continue
        
        rows = ws.get_all_records()
        print(f"Scanning {title} ({len(rows)} rows)...")

        for r in rows:
            # Expected headers: ts, label, confidence, ai_reason, human_label, human_reason, e1_key...
            # Note: get_all_records uses header row. monitor-sheets headers are:
            # ts, label, confidence, ai_reason, human_label, human_reason
            
            human = str(r.get("human_label", "")).strip().upper()
            ai = str(r.get("label", "")).strip().upper()
            
            if human and human != ai and human in ["ON-TASK", "OFF-TASK"]:
                # Reconstruct context
                events = []
                # Check e1_key to e5_key
                for k in range(1, 6):
                    val = r.get(f"e{k}_key", "")
                    if val:
                        events.append(str(val))
                
                mistakes.append({
                    "date": str(d),
                    "ts": r.get("ts"),
                    "ai_label": ai,
                    "human_label": human,
                    "human_reason": r.get("human_reason", ""),
                    "ai_reason": r.get("ai_reason", ""),
                    "events": events
                })
    return mistakes

def optimize_prompt(current_prompt: str, mistakes: List[Dict], prompt_type: str) -> str:
    if not mistakes:
        print(f"No mistakes found for {prompt_type}. Skipping.")
        return current_prompt

    mistakes_text = ""
    for idx, m in enumerate(mistakes, 1):
        mistakes_text += (
            f"\n{idx}. Events: {m['events']}\n"
            f"   AI Said: {m['ai_label']} ({m['ai_reason']})\n"
            f"   Human Said: {m['human_label']} ({m['human_reason']})\n"
        )

    meta_prompt = (
        f"You are an expert prompt engineer optimizing a system for productivity monitoring.\n"
        f"The current prompt is provided below. It failed on the following examples.\n\n"
        f"Mistakes to fix:\n{mistakes_text}\n\n"
        f"Current Prompt:\n```\n{current_prompt}\n```\n\n"
        f"Task: Rewrite the prompt to improve accuracy on these cases WITHOUT overcorrecting or breaking general rules.\n"
        "Keep the same general structure and placeholders (e.g. [insert here]).\n"
        "Do not make it too long.\n"
        "Output ONLY the new prompt content."
    )

    t0 = time.time()
    try:
        resp = openai.chat.completions.create(
            model=MODEL_OPTIMIZER,
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=0.2
        )
        new_prompt = resp.choices[0].message.content.strip()
        # Strip markdown code blocks if present
        if new_prompt.startswith("```"):
            new_prompt = new_prompt.strip("`").strip()
            if new_prompt.startswith("plaintext") or new_prompt.startswith("markdown"):
                 new_prompt = new_prompt.split("\n", 1)[1]
        
        print(f"Optimized {prompt_type} in {time.time()-t0:.2f}s")
        return new_prompt
    except Exception as e:
        print(f"Optimization failed: {e}")
        return current_prompt

def main():
    print("Connecting to Sheets...")
    gc, sh = get_client()
    
    mistakes = fetch_mistakes(sh)
    print(f"Found {len(mistakes)} mistakes.")
    if not mistakes:
        return

    # Update Main Prompt
    print("Reading main prompt...")
    with open(PROMPT_MAIN_FILE, "r", encoding="utf-8") as f:
        main_p = f.read()
    
    new_main = optimize_prompt(main_p, mistakes, "MAIN")
    if new_main != main_p:
        with open(PROMPT_MAIN_FILE, "w", encoding="utf-8") as f:
            f.write(new_main)
        print("Updated prompt-main.txt")

    # Update Critic Prompt
    # (Optional: Filter mistakes where critic was involved or just feed all? For now feed all)
    print("Reading critic prompt...")
    with open(PROMPT_CRITIC_FILE, "r", encoding="utf-8") as f:
        critic_p = f.read()

    new_critic = optimize_prompt(critic_p, mistakes, "CRITIC")
    if new_critic != critic_p:
        with open(PROMPT_CRITIC_FILE, "w", encoding="utf-8") as f:
            f.write(new_critic)
        print("Updated prompt-critic.txt")

    # Cleanup
    if os.path.exists(SERVICE_ACCOUNT_JSON):
        os.remove(SERVICE_ACCOUNT_JSON)

if __name__ == "__main__":
    main()
