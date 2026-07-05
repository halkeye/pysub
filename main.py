import logging
import os
from datetime import timedelta
from string import Template

import configargparse
import platformdirs
import pycountry
import requests
import srt
from faster_whisper import WhisperModel
from moviepy import VideoFileClip
from openai import OpenAI
from secret_type import secret
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

# Suppress OpenAI client HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


def get_language_code(language_name):
    try:
        # Search for the language by its human-readable name
        lang = pycountry.languages.lookup(language_name)

        # Return 2-letter code if available; otherwise fall back to 3-letter code
        return lang.alpha_2 if hasattr(lang, "alpha_2") else lang.alpha_3
    except LookupError:
        return "Language not found"


def get_language_name(code):
    try:
        lang = (
            pycountry.languages.get(alpha_2=code)
            or pycountry.languages.get(alpha_3=code)
            or pycountry.languages.get(name=code)
        )
        if lang is None:
            raise LookupError(f"Language code '{code}' not found.")
        return lang.name.lower()
    except LookupError:
        return "Language not found"


def extract_audio(video_path, audio_path="temp_audio.mp3"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    clip.close()
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

        return translate_with_ollama(
            text,
            source_language,
            target_language,
            model=ollama_model,
            server=ollama_server,
            prompt=prompt,
        )

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
    video_path,
    srt_path,
    target_language=None,
    source_language=None,
    api_key=None,
    provider=None,
    model=None,
    server=None,
    whisper_model=None,
):  # pylint: disable=too-many-locals,too-many-statements

    if srt_path is None:
        srt_path = f"{os.path.splitext(video_path)[0]}.{get_language_code(target_language)}.srt"

    logger.info("Starting subtitle generation... %s", srt_path)

    logger.info("loading whisper model %s", whisper_model)

    audio_path = extract_audio(video_path)

    srt_index = 1
    last_english = None

    whisper_model_object = WhisperModel(
        whisper_model, device="cuda", compute_type="float16"
    )
    with (open(srt_path, "w", encoding="utf-8") as srt_file,):
        # TODO - this audio could just be binaryio, so no writing to disk
        segments, info = whisper_model_object.transcribe(
            audio_path,
            language=(get_language_code(source_language) if source_language else None),
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        if source_language is None:
            source_language = get_language_name(info.language)

        last_end_time = 0.0
        with tqdm(
            total=info.duration,
            desc="Lines",
            unit="line",
            position=1,
            leave=False,
        ) as segment_bar:

            for segment in segments:
                content = english = segment.text.strip()
                if english == last_english:
                    segment_bar.update(segment.end - last_end_time)
                    last_end_time = segment.end
                    continue
                last_english = english

                if source_language.lower() != target_language.lower():
                    content = translate_text(
                        english,
                        source_language,
                        target_language,
                        api_key,
                        provider,
                        ollama_model=model,
                        ollama_server=server,
                    )

                start = timedelta(seconds=segment.start)
                end = timedelta(seconds=segment.end)

                subtitle = srt.Subtitle(
                    index=srt_index,
                    start=start,
                    end=end,
                    content=content.replace("\n", "\\n"),
                )
                srt_file.write(srt.compose([subtitle]))
                srt_index += 1
                segment_bar.update(segment.end - last_end_time)
                last_end_time = segment.end

    logger.info("✅ Subtitles saved to: %s", srt_path)


def main():
    p = configargparse.ArgParser(
        description="Generate subtitles from video(s).",
        config_file_parser_class=configargparse.TomlConfigParser(["pysub"]),
        default_config_files=[
            platformdirs.user_config_path("pysub") / "config.toml",
        ],
    )

    p.add_argument("--config", is_config_file=True, help="config file path")

    p.add_argument("input", help="Path to a video file or directory")
    p.add_argument(
        "--srt_filename",
        help="SRT Filename (default will be video.lang.srt)",
    )
    p.add_argument(
        "--source_language",
        help="Source language for translation",
    )
    p.add_argument(
        "--target_language",
        help="Target language for translation",
        default="english",
    )
    p.add_argument(
        "--api_key",
        type=secret,
        help="API key for translation service",
    )
    p.add_argument(
        "--provider",
        help="Translation provider (openai or ollama)",
        default="ollama",
    )
    p.add_argument(
        "--model",
        help="Model for translation",
        default="translategemma:4b-it-q4_K_M",
    )
    p.add_argument(
        "--server",
        help="Server for Ollama translation",
        default="http://localhost:11434",
    )
    p.add_argument(
        "--whisper_model",
        help="Whisper model to use for transcription",
        default="large-v2",
    )

    logging.basicConfig(
        # TODO - make logging level and file configurable, i think with --log-leve
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("pysub.log"), logging.StreamHandler()],
    )

    args = p.parse_args()

    if args.input == "server":
        logger.info("Config: %s", args)
        raise ValueError("unimplemented")
    else:
        with logging_redirect_tqdm():
            logger.info("Config: %s", args)
            process_single_video(
                args.input,
                args.srt_filename,
                target_language=args.target_language,
                source_language=args.source_language,
                api_key=args.api_key,
                provider=args.provider,
                model=args.model,
                server=args.server,
                whisper_model=args.whisper_model,
            )


if __name__ == "__main__":
    main()
