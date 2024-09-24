# Open Sourced NoteBookLM

![Podcast Creator](https://github.com/mehdihosseinimoghadam/open-sourced-nootbookLM/blob/main/img.png)


## Overview

The Podcast Creator script is designed to automate the process of creating a podcast from a PDF document. It extracts text from the PDF, generates a detailed podcast script using OpenAI's GPT-4 model, converts the script to audio, and then combines the audio with images of the PDF pages to create a video. The final output includes both an audio file and a video file with synchronized audio.



## Examples

### Mistral 7B

[![Watch the video](https://img.youtube.com/vi/sDwxJx8WX3w/0.jpg)](https://www.youtube.com/watch?v=sDwxJx8WX3w&start=26)

<iframe width="560" height="315" src="https://www.youtube.com/embed/K_7kt5_x-Ow?start=26" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


<iframe width="560" height="315"
src="https://www.youtube.com/embed/MUQfKFzIOeU" 
frameborder="0" 
allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" 
allowfullscreen></iframe>

### LLama2


<iframe width="560" height="315" src="https://www.youtube.com/embed/sDwxJx8WX3w?start=26" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


### Attention is all you need

<iframe width="560" height="315" src="https://www.youtube.com/embed/M61t5CXCKtI?start=26" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


## Features

1. **PDF Text Extraction**: Extracts text content from a PDF document.
2. **Script Generation**: Uses OpenAI's GPT-4 model to generate a detailed podcast script based on the extracted text.
3. **Text-to-Speech Conversion**: Converts the generated script into audio using OpenAI's text-to-speech capabilities.
4. **Audio Processing**: Processes the audio to ensure it meets the desired specifications (e.g., stereo, sample rate).
5. **Video Creation**: Converts PDF pages to images and combines them with the audio to create a video.
6. **Environment Configuration**: Loads environment variables from a `.env` file for secure API key management.

## Workflow

1. **Extract Text from PDF**: The script starts by extracting text from the provided PDF file.
2. **Generate Podcast Script**: The extracted text is used to generate a podcast script featuring two hosts, Alice and John, who engage in a detailed conversation about the content.
3. **Convert Script to Audio**: The script is converted to audio, with different voices assigned to Alice and John.
4. **Process Audio**: The audio is processed to ensure it is in the correct format and quality.
5. **Create Video**: Images of the PDF pages are created and combined with the audio to produce a video.
6. **Save Outputs**: The final audio and video files are saved to the specified output paths.

## Usage

To run the project:

1.

```bash
   pip install poetry
```

2.

```bash
   poetry install
```

3. fill .env file
```bash
  OPENAI_API_KEY=""
```


4.

```bash
cd podcast_creator
```

5. To use the script, simply provide the path to the PDF file and run the script. The script will handle the rest, generating the podcast script, converting it to audio, processing the audio, and creating the video.


```python
if name == "main":
pdf_path = "/path/to/your/pdf/document.pdf"
create_podcast_from_pdf(pdf_path)
```




6.

```bash
   poetry run python podcast_creator/main.py
```





## Dependencies

- `langchain`
- `pydantic`
- `openai`
- `pydub`
- `fitz` (PyMuPDF)
- `numpy`
- `subprocess`
- `tqdm`
- `PIL` (Pillow)
- `textwrap`
- `dotenv`

Ensure all dependencies are installed before running the script.



## Conclusion

The Podcast Creator script provides a comprehensive solution for converting PDF documents into engaging podcast episodes, complete with audio and video outputs. By leveraging advanced AI models and audio processing techniques, it automates the entire workflow, making it easy to create high-quality podcast content from textual documents.


## License

Let's Have a Chat ;)
