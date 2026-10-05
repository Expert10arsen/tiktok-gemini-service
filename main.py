import os
import tempfile
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yt_dlp
from google import genai
from google.genai import types

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=GEMINI_API_KEY)


class VideoRequest(BaseModel):
    url: str
    prompt: str = (
        "Подробно опиши, что происходит в этом видео, и перескажи речь."
    )


def download_tiktok_video(url: str, output_path: str):
    ydl_opts = {
        "outtmpl": output_path,
        "format": "mp4/bestvideo+bestaudio/best",
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Ошибка скачивания: {str(e)}"
        )


@app.post("/analyze-tiktok")
def analyze_tiktok(req: VideoRequest):
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, "video.mp4")
        download_tiktok_video(req.url, video_path)

        uploaded_file = client.files.upload(
            file=video_path,
            config=types.UploadFileConfig(mime_type="video/mp4"),
        )

        try:
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)

            if uploaded_file.state.name == "FAILED":
                raise HTTPException(
                    status_code=500, detail="Ошибка обработки файла Gemini"
                )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[uploaded_file, req.prompt],
            )

            return {"success": True, "analysis": response.text}
        finally:
            client.files.delete(name=uploaded_file.name)
