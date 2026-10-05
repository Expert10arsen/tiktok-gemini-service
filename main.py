import os
import tempfile
import time
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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


def download_tiktok_direct(url: str, output_path: str):
    """Получение прямой ссылки через TikWM API и сохранение файла."""
    api_endpoint = "https://www.tikwm.com/api/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    }

    try:
        res = requests.post(
            api_endpoint, data={"url": url, "hd": 1}, headers=headers, timeout=15
        )
        res_data = res.json()

        if res_data.get("code") != 0 or "data" not in res_data:
            raise Exception(res_data.get("msg", "Не удалось извлечь видео"))

        video_download_url = res_data["data"].get(
            "hdplay"
        ) or res_data["data"].get("play")
        if not video_download_url:
            raise Exception("Ссылка на видеопоток не найдена")

        # Скачиваем сам бинарник видео
        video_stream = requests.get(
            video_download_url, stream=True, timeout=30
        )
        with open(output_path, "wb") as f:
            for chunk in video_stream.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка загрузки через API: {str(e)}",
        )


@app.post("/analyze-tiktok")
def analyze_tiktok(req: VideoRequest):
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, "video.mp4")
        download_tiktok_direct(req.url, video_path)

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
