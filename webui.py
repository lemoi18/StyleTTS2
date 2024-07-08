import asyncio
import json
import subprocess
import shlex

import gradio as gr
import os
import torch
import time
import yaml
import torchaudio
import random
import numpy as np
import librosa
from munch import Munch
from phonemizer.backend import EspeakBackend

from StyleTTS2.text_utils import TextCleaner
from utils import *
from Reader import EpubReader

# Path to the settings file
SETTINGS_FILE_PATH = "Configs/generate_settings.yaml"
GENERATE_SETTINGS = {}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
global_phonemizer = None
model = None
model_params = None
sampler = None
textcleaner = None
to_mel = None



def useReader(voice):
    try:

        config_paths= []
        model_paths = []
        for i in voice:
            config_path = get_model_configuration(i)
            config_paths.append(config_path)
            model_path = load_voice_model(i)
            model_paths.append(model_path)

        voices, files = get_voice_list2()

        print(config_paths,model_paths,files, voice)
        return EpubReader(config_paths,model_paths,files, voices)

    except Exception as e:
        print(f"Error : {e}")
        raise




def load_all_models(voice):
    global global_phonemizer, model, model_params, sampler, textcleaner, to_mel
    try:
        print(f"Loading models for voice: {voice}")

        start_time = time.time()
        config = load_configurations(get_model_configuration(voice))
        print(f"Loaded configuration: {config}")

        model_path = load_voice_model(voice)
        print(f"Loaded model path: {model_path}")

        sigma_value = config['model_params']['diffusion']['dist']['sigma_data']
        print(f"Sigma value: {sigma_value}")

        model, model_params = load_models_webui(config, sigma_value, device)
        print("Loaded models with parameters")

        global_phonemizer = load_phonemizer()
        print("Loaded phonemizer")

        sampler = create_sampler(model)
        print("Created sampler")

        textcleaner = TextCleaner()
        print("Initialized text cleaner")

        to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        print("Initialized MelSpectrogram")

        load_pretrained_model(model, model_path=model_path)
        print("Loaded pretrained model")

        end_time = time.time()
        print(f"Models loaded successfully in {end_time - start_time} seconds")
    except Exception as e:
        print(f"Error loading models: {e}")


def get_file_path(root_path, voice, file_extension, error_message):
    try:
        print(f"Getting file path for voice: {voice}, extension: {file_extension}")
        model_path = os.path.join(root_path, voice)
        if not os.path.exists(model_path):
            raise gr.Error(f'No {file_extension} located in "{root_path}" folder')

        for file in os.listdir(model_path):
            if file.endswith(file_extension):
                file_path = os.path.join(model_path, file)
                print(f"Found file: {file_path}")
                return file_path

        raise gr.Error(error_message)
    except Exception as e:
        print(f"Error getting file path: {e}")
        raise


def get_model_configuration(voice):
    try:
        print(f"Getting model configuration for voice: {voice}")
        return get_file_path(root_path="Models", voice=voice, file_extension=".yml",
                             error_message="No configuration for Model specified located")
    except Exception as e:
        print(f"Error getting model configuration: {e}")
        raise


def load_voice_model(voice):
    try:
        print(f"Loading voice model for voice: {voice}")
        return get_file_path(root_path="Models", voice=voice, file_extension=".pth",
                             error_message="No TTS model found in specified location")
    except Exception as e:
        print(f"Error loading voice model: {e}")
        raise


import os

def get_voice_list2(dir=get_voice_dir(), append_defaults=False,
                   extensions=["wav", "mp3", "flac", "pth", "opus", "m4a", "webm", "mp4"]):
    try:
        defaults = []
        os.makedirs(dir, exist_ok=True)
        res = []
        allfiles = {}

        for name in os.listdir(dir):
            if name in defaults:
                continue
            if not os.path.isdir(f'{dir}/{name}'):
                continue
            if len(os.listdir(os.path.join(dir, name))) == 0:
                continue
            files = get_voice(name, dir=dir, extensions=extensions)
            if files:
                res.append(name)
                allfiles[name] = files
            else:
                for subdir in os.listdir(f'{dir}/{name}'):
                    if not os.path.isdir(f'{dir}/{name}/{subdir}'):
                        continue
                    files = get_voice(f'{name}/{subdir}', dir=dir, extensions=extensions)
                    if len(files) == 0:
                        continue
                    res.append(f'{name}/{subdir}')

        res = sorted(res)


        return res, allfiles
    except Exception as e:
        print(f"Error getting voice list: {e}")
        raise



def generate_audio(text, voice, reference_audio_file, seed, alpha, beta, diffusion_steps, embedding_scale,
                   voices_root="voices"):
    try:
        global reader  # Declare that we are using the global reader

        #print(f"Generating audio for text: {text} with voice: {voice}")
        original_seed = int(seed)
        reference_audio_path = os.path.join(voices_root, voice, reference_audio_file)
        reference_dicts = {f'{voice}': f"{reference_audio_path}"}


        start = time.time()
        if original_seed == -1:
            seed_value = random.randint(0, 2 ** 32 - 1)
        else:
            seed_value = original_seed
        set_seeds(seed_value)
        for k, path in reference_dicts.items():
            print(f"here is text: {text}, alpha :{alpha}, beta :{beta}, diffusion_steps :{diffusion_steps}, embedding_scale :{embedding_scale}")
            wav1 = reader.read_passage(voice,reference_audio_file,text,alpha,beta,diffusion_steps,embedding_scale)
            rtf = (time.time() - start)
            print(f"RTF = {rtf:5f}")
            print(f"{k} Synthesized audio path: {path}")
            from scipy.io.wavfile import write
            os.makedirs("results", exist_ok=True)
            audio_opt_path = os.path.join("results", f"{voice}_output.wav")
            write(audio_opt_path, 24000, wav1)

        # Save the settings after generation
        save_settings({
            "text": text,
            "voice": voice,
            "reference_audio_file": reference_audio_file,
            "seed": seed_value if original_seed == -1 else original_seed,
            "alpha": alpha,
            "beta": beta,
            "diffusion_steps": diffusion_steps,
            "embedding_scale": embedding_scale
        })
        return audio_opt_path, [[seed_value]]
    except Exception as e:
        print(f"Error generating audio: {e}")
        raise


def train_model(data):
    try:
        print(f"Training model with data: {data}")
        return f"Model trained with data: {data}"
    except Exception as e:
        print(f"Error training model: {e}")
        raise


def update_settings(setting_value):
    try:
        print(f"Updating settings to: {setting_value}")
        return f"Settings updated to: {setting_value}"
    except Exception as e:
        print(f"Error updating settings: {e}")
        raise


def get_reference_audio_list(voice_name, root="voices"):
    try:
        reference_directory_list = os.listdir(os.path.join(root, voice_name))
        print(f"Reference audio list for voice {voice_name}: {reference_directory_list}")
        return reference_directory_list
    except Exception as e:
        print(f"Error getting reference audio list: {e}")
        raise




def update_voice_settings(voice):
    try:
        print(f"Updating voice settings for: {voice}")
        gr.Info("Wait for models to load...")
        ref_aud_path = update_reference_audio(voice)
        gr.Info("Models finished loading")
        return ref_aud_path
    except Exception as e:
        print(f"Error while updating voice settings: {e}")
        gr.Warning("No models found for the chosen voice, new models not loaded")
        return update_reference_audio(voice)


def load_settings():
    try:
        if not os.path.exists(SETTINGS_FILE_PATH):
            print("Settings file not found, using default settings")
            return get_default_settings()

        with open(SETTINGS_FILE_PATH, "r") as f:
            content = f.read().strip()
            if not content:
                print("Settings file is empty, using default settings")
                return get_default_settings()

            settings = yaml.safe_load(content)
            if settings is None:
                print("Settings file is empty, using default settings")
                return get_default_settings()

            print("Settings loaded successfully")
            return settings

    except FileNotFoundError:
        print("Settings file not found, using default settings")
        return get_default_settings()
    except Exception as e:
        print(f"Error loading settings: {e}")
        raise


def get_default_settings():
    global voice_list_with_defaults, files
    return {
        "text": "",
        "voice": voice_list_with_defaults[0],
        "reference_audio_file": next(iter(files.values()))[0],  # Assuming `files` is a dict of lists
        "seed": "-1",
        "alpha": 0.3,
        "beta": 0.7,
        "diffusion_steps": 30,
        "embedding_scale": 1.0
    }

def save_settings(settings):
    try:
        with open(SETTINGS_FILE_PATH, "w") as f:
            yaml.safe_dump(settings, f)
        print("Settings saved successfully")
    except Exception as e:
        print(f"Error saving settings: {e}")
        raise



def get_audio_duration(audio_path):
    try:
        duration = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
        ).decode('utf-8').strip()
        return float(duration)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Output: {e.output.decode('utf-8')}")
        return None

# Function to create black video
def create_black_video(video_path, duration, video_size=(1280, 720), fps=24):
    if duration is None:
        print("Invalid audio duration. Cannot create black video.")
        return
    width, height = video_size
    command = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'color=c=black:s={width}x{height}:d={duration}',
        '-r', str(fps),
        '-y',  # Overwrite output file if it exists
        video_path
    ]
    subprocess.run(command, check=True)
    print(f"Black video created at {video_path}")



def convert_whisperx_to_ass(whisperx_json_path, ass_path):
    def seconds_to_ass_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds * 100) % 100)  # Correct calculation of centiseconds
        return f"{hours}:{minutes:02}:{secs:02}.{centiseconds:02}"

    header = """[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,1,10,10,10,10,1  # Fontsize increased to 28, Alignment set to 5

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    try:
        with open(whisperx_json_path, "r") as f:
            data = json.load(f)

        with open(ass_path, "w", encoding='utf-8') as f:
            f.write(header)
            for i, segment in enumerate(data['segments']):
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].replace('\\', '\\\\').replace('{', '{{').replace('}', '}}')

                start_timestamp = seconds_to_ass_timestamp(start_time)
                end_timestamp = seconds_to_ass_timestamp(end_time)

                # Adding the highlight effect
                f.write(f"Dialogue: 0,{start_timestamp},{end_timestamp},Default,,0,0,0,,{{\\an5\\bord2\\shad1\\c&H00FF00&}}{text}\n")
    except Exception as e:
        print(f"Error processing the subtitles: {str(e)}")
# Function to add subtitles using FFmpeg
def add_subtitles_ffmpeg(video_path, subtitle_path, audio_path, output_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-vf', f"ass={subtitle_path}",
        '-c:a', 'aac',              # Encode audio to AAC
        '-strict', 'experimental',  # Some FFmpeg versions require this for AAC
        '-y',                       # Overwrite output file without asking
        output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video with subtitles and audio saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print("FFmpeg failed:", e.stderr.decode())  # Print error output for diagnosis

# Function to wait for the JSON file to be created
async def wait_for_file(file_path, timeout=60):
    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for file: {file_path}")
        await asyncio.sleep(0.5)  # Check every 0.5 seconds

async def process_audio_async(file, voice):
    audio_path = get_full_path(file)
    whisperx_json_path = f"results/{voice}_output.json"
    video_path = "results/black_video.mp4"
    ass_path = "results/subtitles.ass"
    final_output_path = "results/final_output.mp4"

    # Generate the WhisperX JSON (assuming WhisperX command is already available in the system)
    subprocess.run(['whisperx', audio_path, '--model', 'large', '--align_model', 'WAV2VEC2_ASR_LARGE_LV60K_960H', '--highlight_words', 'True', '--output_dir', 'results'])

    # Wait for the WhisperX JSON file to be created
    await wait_for_file(whisperx_json_path)

    # Convert WhisperX output to ASS subtitles
    convert_whisperx_to_ass(whisperx_json_path, ass_path)

    # Get the duration of the audio file
    audio_duration = get_audio_duration(audio_path)
    if audio_duration is not None:
        create_black_video(video_path, audio_duration)

    # Add subtitles and audio to the video
    add_subtitles_ffmpeg(video_path, ass_path, audio_path, final_output_path)

    # Return the final video path for playback
    return final_output_path

# Wrapper function to run async function
def process_audio(file, voice):
    return asyncio.run(process_audio_async(file, voice))
# Load models with the default or loaded settings
initial_settings = load_settings()
#voice = initial_settings["voice"]
#if initial_settings["voice"] == None:
# Function to get the full path of the reference audio file
def get_full_path(reference_audio_file):
    # Ensure reference_audio_file does not already include the voices directory
    if reference_audio_file.startswith('./'):
        reference_audio_file = reference_audio_file[2:]
    full_path = os.path.join(reference_audio_file)
    print(f"Full path constructed: {full_path}")  # Debugging information
    return full_path

# Function to update the reference audio player
def update_reference_audio(file):
    full_path = get_full_path(file)
    print(f"Updating reference audio to: {full_path}")  # Debugging information

    # Verify the file exists
    if not os.path.exists(full_path):
        print(f"File does not exist: {full_path}")
        return None

    # Load the audio file using torchaudio
    waveform, sample_rate = torchaudio.load(full_path)
    audio_np = waveform.numpy()
    print(f"Audio shape: {audio_np.shape}, Sample rate: {sample_rate}")  # Debugging information

    return full_path

# Function to update interface based on the selected voice
def update_interface(voice):
    voice_list_with_defaults, files_dict = get_voice_list2(append_defaults=True)
    ref_audio_files = files_dict.get(voice, [])
    if not ref_audio_files:
        return gr.update(choices=[], value=None), None  # No reference audio files available

    first_audio_file_path = get_full_path(ref_audio_files[0])
    print(f"Updating interface for voice: {voice}")  # Debugging information
    return gr.update(choices=ref_audio_files, value=ref_audio_files[0]), first_audio_file_path

# Initialize settings
voice_list_with_defaults, files_dict = get_voice_list2(append_defaults=True)
reader = useReader(voice_list_with_defaults)
# Define the Gradio interface
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Generation"):
            with gr.Column():
                with gr.Row():
                    GENERATE_SETTINGS["text"] = gr.Textbox(label="Input Text", value=initial_settings["text"])
                with gr.Row():
                    with gr.Column():
                        GENERATE_SETTINGS["voice"] = gr.Dropdown(
                            choices=voice_list_with_defaults, label="Voice", type="value",
                            value=initial_settings["voice"]
                        )
                        GENERATE_SETTINGS["reference_audio_file"] = gr.Dropdown(
                            choices=files_dict[initial_settings["voice"]], label="Reference Audio",
                            type="value", value=initial_settings["reference_audio_file"]
                        )
                        reference_audio_output = gr.Audio(label="Reference Audio", value=update_reference_audio(initial_settings["reference_audio_file"]))
                    with gr.Column():
                        GENERATE_SETTINGS["seed"] = gr.Textbox(
                            label="Seed", value=initial_settings["seed"]
                        )
                        GENERATE_SETTINGS["alpha"] = gr.Slider(
                            label="alpha", minimum=0, maximum=2.0, step=0.1, value=initial_settings["alpha"]
                        )
                        GENERATE_SETTINGS["beta"] = gr.Slider(
                            label="beta", minimum=0, maximum=2.0, step=0.1, value=initial_settings["beta"]
                        )
                        GENERATE_SETTINGS["diffusion_steps"] = gr.Slider(
                            label="Diffusion Steps", minimum=0, maximum=400, step=1,
                            value=initial_settings["diffusion_steps"]
                        )
                        GENERATE_SETTINGS["embedding_scale"] = gr.Slider(
                            label="Embedding Scale", minimum=0, maximum=4.0, step=0.1,
                            value=initial_settings["embedding_scale"]
                        )
                    with gr.Column():
                        generation_output = gr.Audio(label="Generated Output")
                        seed_output = gr.Dataframe(
                            headers=["Seed"],
                            datatype=["number"],
                            value=[],
                            height=200,
                            min_width=200
                        )
                with gr.Row():
                    generate_button = gr.Button("Generate")

                # Update the dropdown and audio player when the voice is changed
                GENERATE_SETTINGS["voice"].change(fn=update_interface,
                                                  inputs=GENERATE_SETTINGS["voice"],
                                                  outputs=[GENERATE_SETTINGS["reference_audio_file"], reference_audio_output])

                # Update the audio player when the reference audio file is changed
                GENERATE_SETTINGS["reference_audio_file"].change(
                    fn=update_reference_audio,
                    inputs=GENERATE_SETTINGS["reference_audio_file"],
                    outputs=reference_audio_output
                )

                generate_button.click(generate_audio,
                                      inputs=[
                                          GENERATE_SETTINGS["text"],
                                          GENERATE_SETTINGS["voice"],
                                          GENERATE_SETTINGS["reference_audio_file"],
                                          GENERATE_SETTINGS["seed"],
                                          GENERATE_SETTINGS["alpha"],
                                          GENERATE_SETTINGS["beta"],
                                          GENERATE_SETTINGS["diffusion_steps"],
                                          GENERATE_SETTINGS["embedding_scale"]
                                      ],
                                      outputs=[generation_output, seed_output])
                process_button = gr.Button("Process Audio")
                process_output = gr.Video()

                process_button.click(
                    fn=lambda _: process_audio(f"results/{GENERATE_SETTINGS['voice'].value}_output.wav",GENERATE_SETTINGS['voice'].value),
                    inputs=[],
                    outputs=process_output
                )

        with gr.TabItem("Training"):
            training_data = gr.Textbox(label="Enter training data")
            training_output = gr.Textbox(label="Training Output", interactive=False)
            train_button = gr.Button("Train")
            train_button.click(train_model, inputs=training_data, outputs=training_output)

        with gr.TabItem("Settings"):
            settings_input = gr.Textbox(label="Enter setting value")
            settings_output = gr.Textbox(label="Settings Output", interactive=False)
            settings_button = gr.Button("Update Settings")
            settings_button.click(update_settings, inputs=settings_input, outputs=settings_output)

# Launch the interface
demo.launch()