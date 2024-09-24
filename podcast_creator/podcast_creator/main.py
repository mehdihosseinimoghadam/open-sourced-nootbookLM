from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import json
import os
import openai
from pydub import AudioSegment
import fitz  # PyMuPDF
import numpy as np
import subprocess
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import textwrap
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def create_podcast_from_pdf(
    pdf_path,
    output_audio_file="podcast_output.mp3",
    output_video_file="output_video_with_audio.mp4",
):
    # Set your OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Extract text content from PDF
    def extract_text_from_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    content = extract_text_from_pdf(pdf_path)

    # Define the output structure
    class PodcastSegment(BaseModel):
        speaker: str = Field(description="The name of the speaker (Alice or John)")
        text: str = Field(description="The text spoken by the speaker")

    class PodcastScript(BaseModel):
        segments: List[PodcastSegment]

    # Create the output parser
    parser = PydanticOutputParser(pydantic_object=PodcastScript)

    # Create the ChatGPT model
    chat = ChatOpenAI(temperature=0.5, model_name="gpt-4o-mini")

    # Define the prompt template
    template = """
    You are an expert podcast script writer responsible for creating an extended, highly detailed, and thorough script for a podcast episode titled “Vox AI News.” This podcast features two hosts, Alice and John, who will engage in an in-depth, conversational dialogue while covering the content provided.

    Your task is to write a very long, highly detailed script where Alice and John:

        • Engage in a natural conversation that explains the subject in great detail.
        • Provide full and comprehensive analysis of the content, ensuring all aspects of the subject are covered.
        • Make sure each point from the content is explained fully, with examples and contextual explanations where needed, to ensure the audience has a complete understanding of the subject.

    Key instructions:

        1. The script must be long and thorough, focusing on delivering an exhaustive explanation of the content. Include additional context, background information, and potential implications of the subject where necessary.
        2. Alice and John should introduce the topic, greet each other warmly, and maintain a friendly, engaging tone throughout the conversation.
        3. They should go deep into the subject, offering multiple layers of explanation. For instance, if the content covers a complex topic, they should break it down, give examples, and analyze how it impacts the broader landscape.
        4. The questions should encourage detailed responses, with follow-up inquiries that lead to more exploration of the topic.
        5. After asking each other questions, Alice and John will provide rich, comprehensive answers, making sure to fully unpack the concepts.
        6. The podcast will end with a full summary of the discussion, and Alice and John will close with a cheerful sign-off (e.g., “Thanks for joining us! Stay curious and stay informed! We’ll catch you on the next episode of Vox AI News.”).

    Here’s the content to base the podcast on:

    {content}

    {format_instructions}

    Make sure that Alice and John will cover the topic with at least 40 questions
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Function to generate the podcast script and save it to a JSON file
    def generate_and_save_podcast_script(content, output_file="podcast_script.json"):
        messages = prompt.format_messages(
            content=content, format_instructions=parser.get_format_instructions()
        )

        response = chat(messages)

        try:
            parsed_output = parser.parse(response.content)
            script_json = [segment.dict() for segment in parsed_output.segments]

            # Save the script to a JSON file
            with open(output_file, "w") as f:
                json.dump(script_json, f, indent=2)

            print(f"Podcast script saved to {output_file}")
            return script_json
        except Exception as e:
            print(f"Error parsing output: {e}")
            return response.content

    podcast_script = generate_and_save_podcast_script(content)

    # If you want to print the script as well
    print(json.dumps(podcast_script, indent=2))

    # Define the voices for each speaker
    ALICE_VOICE = "nova"  # Female voice
    JOHN_VOICE = "echo"  # Male voice

    def text_to_speech(text, voice):
        response = openai.audio.speech.create(model="tts-1", voice=voice, input=text)
        return response.content

    def create_podcast_audio(script, output_file):
        full_audio = AudioSegment.empty()

        for segment in script:
            speaker = segment["speaker"]
            text = segment["text"]

            voice = ALICE_VOICE if speaker == "Alice" else JOHN_VOICE

            audio_content = text_to_speech(text, voice)

            # Save the audio content to a temporary file
            with open("temp.mp3", "wb") as f:
                f.write(audio_content)

            # Load the audio segment and append it to the full audio
            audio_segment = AudioSegment.from_mp3("temp.mp3")
            full_audio += audio_segment

        # Load infade and outfade
        infade = AudioSegment.from_wav("./podcast_creator/infade.wav")
        outfade = AudioSegment.from_wav("./podcast_creator/outfade.wav")

        # Calculate overlap durations
        overlap_duration = 7000  # 5 seconds in milliseconds

        # Prepare infade
        infade_duration = len(infade)
        if infade_duration > overlap_duration:
            infade_start = infade[: infade_duration - overlap_duration]
            infade_overlap = infade[-overlap_duration:]
        else:
            infade_start = AudioSegment.silent(
                duration=overlap_duration - infade_duration
            )
            infade_overlap = infade

        # Prepare outfade
        outfade_duration = len(outfade)
        if outfade_duration > overlap_duration:
            outfade_overlap = outfade[:overlap_duration]
            outfade_end = outfade[overlap_duration:]
        else:
            outfade_overlap = outfade
            outfade_end = AudioSegment.silent(
                duration=overlap_duration - outfade_duration
            )

        # Combine all parts
        final_audio = (
            infade_start
            + infade_overlap.overlay(full_audio[:overlap_duration])
            + full_audio[overlap_duration:-overlap_duration]
            + full_audio[-overlap_duration:].overlay(outfade_overlap)
            + outfade_end
        )

        # Export the final audio file
        final_audio.export(output_file, format="mp3")
        print(f"Podcast audio saved as {output_file}")

    create_podcast_audio(podcast_script, output_audio_file)

    def process_audio(input_file, output_file, target_sample_rate=44100):
        # Load the audio file
        audio = AudioSegment.from_mp3(input_file)

        # Convert to stereo if mono
        if audio.channels == 1:
            audio = audio.set_channels(2)

        # Resample to target sample rate
        audio = audio.set_frame_rate(target_sample_rate)

        # Export the processed audio
        audio.export(output_file, format="mp3")

        print(f"Processed audio saved as {output_file}")
        print(f"New sample rate: {audio.frame_rate} Hz")
        print(f"Channels: {audio.channels}")

    # Process the audio
    processed_audio_file = "podcastoutput_processed.mp3"
    process_audio(output_audio_file, processed_audio_file)

    def get_audio_duration(audio_path):
        audio = AudioSegment.from_mp3(audio_path)
        return len(audio) / 1000.0  # Duration in seconds

    def pdf_to_video_with_audio(
        pdf_path, audio_path, output_video_path, fps=30, zoom=2
    ):
        # Get audio duration
        audio_duration = get_audio_duration(audio_path)

        # Open the PDF
        doc = fitz.open(pdf_path)

        # Calculate duration for each page
        page_duration = audio_duration / len(doc)

        # Create a temporary directory for images
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)

        print("Converting PDF pages to images...")
        for page_num in tqdm(range(len(doc))):
            page = doc.load_page(page_num)

            # Increase resolution by using a higher zoom factor
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Save the image
            image_path = os.path.join(temp_dir, f"page_{page_num:03d}.png")
            pix.save(image_path)

        # Create a file with image filenames
        with open("images.txt", "w") as f:
            for page_num in range(len(doc)):
                image_path = os.path.join(temp_dir, f"page_{page_num:03d}.png")
                f.write(f"file '{image_path}'\n")
                f.write(f"duration {page_duration + 10}\n")

        # Use FFmpeg to create a video with audio
        print("Creating video with audio...")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f",
            "concat",  # Use concat demuxer
            "-safe",
            "0",
            "-i",
            "images.txt",  # Input file list
            "-i",
            audio_path,  # Input audio file
            "-c:v",
            "libx264",  # Video codec
            "-preset",
            "slow",  # Encoding preset (slower = better compression)
            "-crf",
            "17",  # Constant Rate Factor (lower = better quality, 17-18 is visually lossless)
            "-c:a",
            "aac",  # Audio codec
            "-b:a",
            "192k",  # Audio bitrate
            "-vf",
            f"fps={fps},format=yuv420p",  # Set frame rate and ensure compatibility
            "-shortest",  # Finish encoding when the shortest input stream ends
            output_video_path,
        ]

        subprocess.run(ffmpeg_cmd, check=True)

        # Clean up temporary files
        for page_num in range(len(doc)):
            os.remove(os.path.join(temp_dir, f"page_{page_num:03d}.png"))
        os.rmdir(temp_dir)
        os.remove("images.txt")

        print(f"Video with audio saved as {output_video_path}")

    # Convert PDF to video with audio
    pdf_to_video_with_audio(
        pdf_path, processed_audio_file, output_video_file, fps=30, zoom=2
    )

    print("Podcast creation process completed successfully!")


# Example usage
if __name__ == "__main__":
    pdf_path = "/Users/mehdihosseinimoghadam/Desktop/temp/open-sourced-nootbookLM/pdf/attention.pdf"
    create_podcast_from_pdf(pdf_path)
