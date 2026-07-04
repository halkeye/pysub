import argparse
import json
import logging
import os
from datetime import timedelta
from string import Template

import jsonschema
import pycountry
import requests
import srt
import whisper
from moviepy import VideoFileClip
from openai import OpenAI
from pydub import AudioSegment
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("pysub.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Suppress OpenAI client HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def get_language_code(language_name):
    try:
        # Search for the language by its human-readable name
        lang = pycountry.languages.lookup(language_name)

        # Return 2-letter code if available; otherwise fall back to 3-letter code
        return lang.alpha_2 if hasattr(lang, "alpha_2") else lang.alpha_3
    except LookupError:
        return "Language not found"


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    schema_path = os.path.join(os.path.dirname(__file__), "schemas/pysub.schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema.validate(instance=config, schema=schema)
    return config


def extract_audio(video_path, audio_path="temp_audio.mp3"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    return audio_path


def translate_text(
    text,
    source_language,
    target_language,
    api_key=None,
    provider="openai",
    ollama_model=None,
    ollama_server=None,
    prompt=None,
):
    if provider == "openai":
        return translate_with_openai(text, source_language, target_language, api_key)

    if provider == "ollama":
        if ollama_model is None:
            raise ValueError(
                "Model must be specified for Ollama translation verification."
            )

        if ollama_server is None:
            raise ValueError(
                "Server must be specified for Ollama translation verification."
            )

        result = translate_with_ollama(
            text,
            source_language,
            target_language,
            model=ollama_model,
            server=ollama_server,
            prompt=prompt,
        )
        return result
        # return verify_or_retranslate_ollama(
        #     text,
        #     result,
        #     source_language,
        #     target_language,
        #     model=ollama_model,
        #     server=ollama_server,
        #     prompt=prompt,
        # )

    raise ValueError(f"Unsupported provider: {provider}")


def translate_with_openai(text, source_language, target_language, api_key):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a professional translation engine. "
                    f"Translate the following {source_language} sentence into **{target_language}** only. Do not use any other language except the one provide. "
                    f"You must translate from {source_language} to **{target_language}**. Do not confuse one language with another. Check and verify your work after you are done"
                    f"Respond with ONLY the *{target_language} sentence, no extra commentary."
                ),
            },
            {"role": "user", "content": f"{text}"},
        ],
    )
    return response.choices[0].message.content.strip()


def translate_with_ollama(
    text,
    source_language,
    target_language,
    model=None,
    server="http://localhost:11434",
    prompt=None,
):
    if model is None:
        raise ValueError("Model must be specified for Ollama translation verification.")

    if prompt is None:
        prompt = (
            "You are a professional $SOURCE_LANG ($SOURCE_CODE) to $TARGET_LANG ($TARGET_CODE) translator.\n"
            "Your goal is to accurately convey the meaning and nuances of the original $SOURCE_LANG text while adhering to $TARGET_LANG grammar, vocabulary, and cultural sensitivities.\n"
            "\n"
            "🔒 Strict Output Rules:\n"
            "1. The translation MUST be written exclusively in **$TARGET_LANG script**.\n"
            "2. You MUST NOT include:\n"
            "   - Romanized or transliterated text\n"
            "   - $SOURCE_LANG words or phrases\n"
            "   - Commentary, notes, or metadata\n"
            "   - The original $SOURCE_LANG sentence\n"
            "   - Any newlines or other control characters\n"
            "3. The output MUST be clear, fluent, and natural to a native speaker of $TARGET_LANG.\n"
            "4. Your highest priority is accuracy and clarity for native speakers.\n"
            "5. ✅ After translating, **you MUST internally verify** that the output contains only $TARGET_LANG script, punctuation, or peoples names and contains NO $SOURCE_LANG, romanization, or foreign words.\n"
            "6. If verification fails, silently retry the translation until the output is clean and correct.\n\n"
            "\n"
            "Produce only the $TARGET_LANG translation, without any additional plesantries, responses, explanations or commentary. Please translate the following $SOURCE_LANG text into $TARGET_LANG:\n"
            "\n"
            "\n"
            "$TEXT"
        )

    resolved_prompt = Template(prompt).safe_substitute(
        {
            "SOURCE_LANG": source_language,
            "SOURCE_CODE": get_language_code(source_language),
            "TARGET_LANG": target_language,
            "TARGET_CODE": get_language_code(target_language),
            "TEXT": text,
        }
    )

    body = {"model": model, "prompt": resolved_prompt, "stream": False}

    response = requests.post(
        f"{server}/api/generate",
        json=body,
    )

    response.raise_for_status()

    return response.json()["response"].strip().strip('"')


def process_single_video(
    video_path, srt_path, config, parent_bar=None
):  # pylint: disable=too-many-locals,too-many-statements
    translate = config.get("translate", False)
    target_language = config.get("target_language", "english")
    source_language = config.get("source_language", "thai")
    api_key = config.get("api_key")
    provider = config.get("provider", "openai")
    ollama_model = config.get("ollama_model", "translategemma:4b-it-q4_K_M")
    ollama_server = config.get("ollama_server", "http://localhost:11434")
    chunk_duration_sec = config.get("chunk_duration_sec", 60)
    chunk_overlap_sec = config.get("chunk_overlap_sec", 5)

    if srt_path is None:
        srt_path = f"{os.path.splitext(video_path)[0]}.{get_language_code(target_language)}.srt"

    logger.info("Starting subtitle generation... %s", srt_path)

    audio_path = extract_audio(video_path)
    audio = AudioSegment.from_file(audio_path)
    chunk_len = chunk_duration_sec * 1000
    overlap = chunk_overlap_sec * 1000
    chunks = [
        audio[i : i + chunk_len] for i in range(0, len(audio), chunk_len - overlap)
    ]

    srt_index = 1
    last_english = None
    offset = 0.0

    with (
        open(srt_path, "w", encoding="utf-8") as srt_file,
        tqdm(
            total=len(chunks),
            desc=f"Chunks [{os.path.basename(video_path)}]",
            position=1,
            leave=False,
        ) as chunk_bar,
    ):
        for i, chunk in enumerate(chunks):
            chunk_file = f"chunk_{i}.mp3"
            chunk.export(chunk_file, format="mp3")

            try:
                model = whisper.load_model("base")
                result = model.transcribe(chunk_file, language="en")
            except Exception as e:
                logger.error("❌ Failed to transcribe chunk %s: %s", i, e)
                chunk_bar.update(1)
                continue

            segments = result.get("segments", [])
            with tqdm(
                total=len(segments),
                desc=f"Segments [Chunk {i + 1}]",
                position=2,
                leave=False,
            ) as segment_bar:

                for segment in segments:
                    english = segment["text"].strip()
                    if english == last_english:
                        segment_bar.update(1)
                        continue
                    last_english = english

                    start = timedelta(seconds=offset + segment["start"])
                    end = timedelta(seconds=offset + segment["end"])

                    try:
                        content = (
                            translate_text(
                                english,
                                source_language,
                                target_language,
                                api_key,
                                provider,
                                ollama_model,
                                ollama_server,
                            )
                            if translate
                            else english
                        )
                    except Exception as e:
                        content = "[Translation error]"
                        logger.error("[Chunk %d] Translation error: %s", i, e)

                    subtitle = srt.Subtitle(
                        index=srt_index, start=start, end=end, content=content
                    )
                    srt_file.write(srt.compose([subtitle]))
                    srt_index += 1
                    segment_bar.update(1)

            offset += (chunk_len - overlap) / 1000.0
            chunk_bar.update(1)

            try:
                os.remove(chunk_file)
            except Exception:
                logger.warning("⚠️ Could not remove temp chunk file: %s", chunk_file)

    logger.info("✅ Subtitles saved to: %s", srt_path)


def process_video_directory(directory_path, config):
    video_files = [
        f
        for f in os.listdir(directory_path)
        if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))
    ]

    with tqdm(total=len(video_files), desc="Videos", position=0) as video_bar:
        for filename in video_files:
            video_path = os.path.join(directory_path, filename)
            srt_path = os.path.splitext(video_path)[0] + ".srt"
            logger.info("Processing: %s", video_path)
            process_single_video(video_path, srt_path, config, parent_bar=video_bar)
            video_bar.update(1)


def verify_or_retranslate_ollama(
    original_english,
    translated_text,
    source_language,
    target_language,
    model="translategemma:4b-it-q4_K_M",
    max_retries=10,
    server="http://localhost:11434",
    prompt=None,
):
    """Verifies if translation is in the correct script and retries if not."""

    if model is None:
        raise ValueError("Model must be specified for Ollama translation verification.")

    verify_prompt = (
        f"You are a linguistic verification assistant.\n"
        f"Verify if the following sentence is entirely in {target_language} script, punctuation, or peoples names and contains no {source_language}, romanization, or foreign words.\n"
        f"Respond with YES if it is valid {target_language}. If it contains any incorrect elements, respond with NO, followed by two new lines, and details about why its incorrect.\n\n"
        f"---\nSentence:\n{translated_text}\n\nAnswer:"
    )
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{server}/api/generate",
                json={"model": model, "prompt": verify_prompt, "stream": False},
            )
            response.raise_for_status()
            answer = response.json()["response"].strip().lower()
            if "yes" in answer:
                return translated_text

            logger.info(
                "[Verify Attempt %d] Verifying '%s' gave invalid result of '%s' Retrying translation...",
                attempt + 1,
                translated_text,
                answer,
            )

            translated_text = translate_with_ollama(
                original_english,
                source_language,
                target_language,
                model=model,
                server=server,
                prompt=prompt,
            )
        except Exception as e:
            logger.error("Verification error: %s", e)
            break
    return f"[Translation verification failed] - {original_english}"


def main():
    parser = argparse.ArgumentParser(description="Generate subtitles from video(s).")
    parser.add_argument("input", help="Path to a video file or directory")
    parser.add_argument(
        "--srt_output", help="Path to save .srt output or directory", required=False
    )
    parser.add_argument("--config", help="Path to config JSON", required=False)
    args = parser.parse_args()

    with logging_redirect_tqdm():
        config = load_config(args.config) if args.config else {}
        process_single_video(args.input, args.srt_output, config)


if __name__ == "__main__":
    main()
