import os
import json
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini (GenAI SAR)
# NEW GEMINI SDK (CORRECT - 2025+)
from google import genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Gemini Client Initialized Successfully")
    except Exception as e:
        print("❌ Gemini Init Error:", e)
        gemini_client = None
else:
    print("⚠️ GEMINI_API_KEY not found. Using fallback SAR.")


# Import YOUR AML pipeline
from aml_hybrid_system import AMLPipeline

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
OUTPUT_DIR = "aml_outputs"
AUDIT_FILE = os.path.join(OUTPUT_DIR, "audit_trail.json")
RISK_FILE = os.path.join(OUTPUT_DIR, "account_risk_scores.csv")

# -----------------------------------------------------------------------------
# FASTAPI INIT
# -----------------------------------------------------------------------------
app = FastAPI(title="Hybrid AML Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# GLOBAL CACHE (Prevents re-running heavy pipeline every API call)
# -----------------------------------------------------------------------------
PIPELINE_RAN = False
AUDIT_DATA = []
RISK_DATA = None


# -----------------------------------------------------------------------------
# RUN PIPELINE (ONLY ONCE)
# -----------------------------------------------------------------------------
def run_pipeline_once():
    global PIPELINE_RAN, AUDIT_DATA, RISK_DATA

    if PIPELINE_RAN:
        return

    print("\n🚀 Starting AML Pipeline (One-Time Initialization)...")

    # If outputs already exist, LOAD instead of re-running (VERY IMPORTANT)
    if os.path.exists(AUDIT_FILE) and os.path.exists(RISK_FILE):
        print("📂 Loading existing AML outputs...")
        with open(AUDIT_FILE, "r") as f:
            AUDIT_DATA = json.load(f)
        RISK_DATA = pd.read_csv(RISK_FILE)
        PIPELINE_RAN = True
        return

    # Otherwise run the heavy pipeline
    print("⚙️ Running full AML pipeline (this may take time)...")
    pipeline = AMLPipeline()
    results, metrics = pipeline.run(top_k_sar=100)

    # Load generated outputs
    if os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, "r") as f:
            AUDIT_DATA = json.load(f)

    if os.path.exists(RISK_FILE):
        RISK_DATA = pd.read_csv(RISK_FILE)

    PIPELINE_RAN = True
    print("✅ Pipeline ready. Outputs cached.")


# -----------------------------------------------------------------------------
# ROOT HEALTH CHECK
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "Hybrid AML Backend Running"}


# -----------------------------------------------------------------------------
# GET HIGH RISK ACCOUNTS (For Left Panel Table)
# -----------------------------------------------------------------------------
@app.get("/api/high-risk-accounts")
def get_high_risk_accounts():
    run_pipeline_once()

    if RISK_DATA is None:
        return []

    # Sort by anomaly score (highest risk first)
    top_accounts = RISK_DATA.sort_values(
        by="anomaly_score", ascending=False
    ).head(100)

    response = []
    for _, row in top_accounts.iterrows():
        response.append({
            "account_id": str(row.get("account_id")),
            "anomaly_score": float(row.get("anomaly_score", 0)),
            "rule_score": int(row.get("rule_score", 0)),
            "typology": str(row.get("typology", "Unclassified"))
        })

    return response


# -----------------------------------------------------------------------------
# GET FULL AUDIT FINDINGS (Explainability JSON Panel)
# -----------------------------------------------------------------------------
@app.get("/api/account-findings/{account_id}")
def get_account_findings(account_id: str):
    run_pipeline_once()

    for record in AUDIT_DATA:
        if str(record.get("account_id")) == str(account_id):
            return record

    return {"error": "Account not found in audit trail"}


# -----------------------------------------------------------------------------
# GEMINI-POWERED LONG SAR GENERATION (MAIN UPGRADE)
# -----------------------------------------------------------------------------
@app.get("/api/generate-sar/{account_id}")
def generate_sar(account_id: str):
    run_pipeline_once()

    record = None
    for r in AUDIT_DATA:
        if str(r.get("account_id")) == str(account_id):
            record = r
            break

    if not record:
        return {"sar": "No SAR data available for this account."}

    evidence = {
        "account_id": record.get("account_id"),
        "anomaly_score": record.get("anomaly_score"),
        "rule_score": record.get("rule_score"),
        "typology": record.get("typology"),
        "typology_confidence": record.get("typology_confidence"),
        "rule_triggers": record.get("rule_triggers"),
        "top_anomaly_features": record.get("top_anomaly_features"),
        "linked_accounts": record.get("linked_accounts"),
        "transaction_chain": record.get("tx_chain_sample"),
        "statistical_features": record.get("statistical_features"),
        "graph_features": record.get("graph_features"),
    }

    # If Gemini not configured, fallback
    if gemini_client is None:
        return {
            "sar": f"""
Suspicious Activity Report (Fallback)

Account {evidence['account_id']} has been flagged with anomaly score 
{evidence['anomaly_score']} under typology {evidence['typology']}.

Key triggers: {evidence['rule_triggers']}

This account shows anomalous transactional and network behaviour.
Manual compliance review is recommended.
"""
        }

    prompt = f"""
You are a senior AML compliance officer at a global bank.

Write a FULL, detailed Suspicious Activity Report (SAR) based ONLY on the 
forensic evidence below.

STRICT REQUIREMENTS:
- 600–900 words
- Formal regulatory tone (FATF / Banking Compliance)
- Multi-paragraph structured report
- Explain WHY activity is suspicious
- Reference typology, graph behavior, anomaly score
- No hallucination, use only given evidence

Strict formatting rules:
- Do NOT use Markdown
- Do NOT use asterisks (*)
- Do NOT use bold or italic formatting
- Do NOT use headings with symbols like ** or ---
- Do NOT use bullet points or numbered lists with symbols
- Write in formal banking/compliance report style
- Use clear paragraphs and simple numbered sections only (1., 2., 3.)
- Ensure the output looks like an official regulatory report, not formatted markdown

EVIDENCE:
{evidence}

STRUCTURE:
1. Executive Summary
2. Transaction Behaviour Analysis
3. Network & Typology Indicators
4. Risk & Compliance Justification
5. Recommendation for Regulatory Review
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        sar_text = response.text
        return {"sar": sar_text}

    except Exception as e:
        return {"sar": f"Gemini SAR generation failed: {str(e)}"}
