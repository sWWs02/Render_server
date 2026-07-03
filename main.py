"""
Gemini 프록시 서버
- Flutter 앱에서 이미지(base64)를 받아 Gemini API 호출 후 결과 반환
- API 키는 서버 환경변수에만 존재, 앱에는 없음

실행:
  pip install fastapi uvicorn requests python-dotenv
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import requests
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 로딩 (로컬 개발용)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ── 환경변수 ───────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
SERVER_API_KEY = os.environ.get("SERVER_API_KEY", "")  # 서버 자체 인증키 (선택)

EXTRACTION_PROMPT = """
You are a precise OCR system. Analyze the provided image and extract the following information.
Strictly return the output in JSON format only, matching this structure:
{
    "model_no": "Exact Model number (Model no)",
    "serial_no": "Exact Serial number (Serial no)",
    "part_number": "Part number (P/N) value based on the rules below, otherwise null",
    "manufacturing_date": "Manufacturing date or year-month (제조년월)",
    "repair_date": "Previous repair date or year-month (전회수리년월) if exists, otherwise null"
}

FLEXIBLE RULES FOR "part_number":
1. CRITICAL (Standalone P/N): On many labels, the part number stands alone WITHOUT any prefix or anchor words like "P/N" or "PART NO.". Even if there is absolutely no text next to it, scan the entire label to find a standalone string that matches the pattern formats below.
2. The target format patterns to look for anywhere on the label are:
   - Pattern 1: [Alphanumeric]-[6 digits]-[2 digits] (e.g., 2L39-000099-42)
   - Pattern 2: [4 digits]-[5 digits] (e.g., 0190-15323 or 0190-15323B)
3. If you find a string that matches these structures (or looks very close to them with minor blur/errors), extract your BEST GUESS as the "part_number". Do not return null easily.
4. Only return null if there is absolutely no string containing hyphens (-) or any text that could be a part number.
5. NEVER duplicate the "model_no" or "serial_no" into the "part_number" field.

CRITICAL RULES FOR "manufacturing_date" AND "repair_date":
1. The format MUST strictly be "YYYY.MM" (e.g., 2010.10).
2. Remove all spaces within the date string.
3. If the month is a single digit, pad it with a leading zero (e.g., convert "2010.1" to "2010.01").
4. If the date uses separators like "/" or "-", always convert them to "." (e.g., convert "2010/10" to "2010.10").

Do not include any conversational text or markdown code blocks (like ```json). Just return the raw JSON string.
"""


class ExtractRequest(BaseModel):
    image: str        # base64 인코딩된 JPEG
    mime_type: str = "image/jpeg"


@app.get("/health")
def health():
    return {"status": "ok", "model": GEMINI_MODEL}


@app.post("/extract")
def extract(
    req: ExtractRequest,
    x_api_key: str = Header(default=""),
):
    # 서버 인증키 검사 (SERVER_API_KEY 설정 시)
    if SERVER_API_KEY and x_api_key != SERVER_API_KEY:
        raise HTTPException(status_code=401, detail="인증 실패")

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="서버에 GEMINI_API_KEY 미설정")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": EXTRACTION_PROMPT.strip()},
                {"inline_data": {"mime_type": req.mime_type, "data": req.image}},
            ]
        }],
        "generationConfig": {"responseMimeType": "application/json"},
    }

    try:
        res = requests.post(url, json=payload, timeout=45)
        res.raise_for_status()
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Gemini 오류: {e} — {res.text[:300]}")
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Gemini 응답 시간 초과")

    text = res.json()["candidates"][0]["content"]["parts"][0]["text"]
    return {"text": text}
