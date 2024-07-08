# requires ebooklib bs4 nltk tqdm
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import nltk
import numpy as np
from Core import ExampleApplicationsCore
import torch
class EpubReader:
    class StyleCache:
        def __init__(self):
            self.styles = {}

        def add_style(self, key, style):
            self.styles[key] = style

        def get_style(self, key):
            return self.styles.get(key)

    def __init__(self, config_paths, model_paths, files_dict, voices):
        if not isinstance(config_paths, list):
            config_paths = [config_paths]
        if not isinstance(model_paths, list):
            model_paths = [model_paths]
        if not isinstance(voices, list):
            voices = [voices]

        self.cores = {}
        self.styles = {}
        self.style_cache = self.StyleCache()

        for voice, config_path, model_path in zip(voices, config_paths, model_paths):
            print(f"Loading model for voice {voice} from {config_path} and {model_path}")
            core = ExampleApplicationsCore()
            core.load_model(config_path, model_path)
            self.cores[voice] = core
            print(f"Done loading model for voice {voice} from {config_path} and {model_path}")

            ref_audio_paths = files_dict.get(voice, [])
            if not ref_audio_paths:
                raise ValueError(f"No reference audio files found for voice: {voice}")
            for ref_audio_path in ref_audio_paths:
                style = self.set_style(core, ref_audio_path)
                self.style_cache.add_style(ref_audio_path, style)
            self.styles[voice] = ref_audio_paths[0]

        print(f"Initialized cores: {self.cores.keys()}")
        print(f"Initialized styles: {self.styles.keys()}")

    def set_style(self, core, ref_audio_path):
        print(f"Calculating style for {ref_audio_path}")
        style = core.style_from_path(ref_audio_path)
        print(f"Done calculating style for {ref_audio_path}")
        return style

    def read_sentences(self, core, style, sentences, alpha=0.1, beta=0.9, t=0.7, diffusion_steps=5, embedding_scale=1, target_wpm=150):
        s_prev = None
        wavs = []
        for sentence in tqdm(sentences, desc="Inferring audio."):
            sentence = sentence.lower()
            wav, s_prev = core.LFinference(sentence, s_prev, style, alpha=alpha, beta=beta, t=t, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale)
            wavs.append(wav)
        return np.concatenate(wavs)

    def read_passage(self, voice, reference_audio_file, p, alpha, beta, diffusion_steps, embedding_scale, **kwargs):
        core = self.cores.get(voice)
        style_path = self.styles.get(voice)
        if not core or not style_path:
            raise ValueError(f"Core or style not found for voice: {voice}")
        texts = nltk.sent_tokenize(p)
        style = self.style_cache.get_style(style_path)
        if style is None or (isinstance(style, torch.Tensor) and style.nelement() == 0):
            raise ValueError(f"Style not found or is empty for path: {style_path}")
        return self.read_sentences(core, style, texts, alpha, beta, diffusion_steps=diffusion_steps, embedding_scale=embedding_scale, **kwargs)

    def read_epubs(self, paths, voice, **kwargs):
        if not isinstance(paths, list):
            paths = [paths]

        all_chapter_audios = []

        for i, path in enumerate(paths):
            assert os.path.exists(path), f"Path {path} does not exist"
            book = epub.read_epub(path)
            chapters = []
            chapter_audios = []

            core = self.cores.get(voice)
            style_path = self.styles.get(voice)
            if not core or not style_path:
                raise ValueError(f"Core or style not found for voice: {voice}")

            for chapter in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                soup = BeautifulSoup(chapter.content, 'html.parser')
                text = soup.body.get_text(separator=' ').strip().replace('\n', '.')
                if len(text) > 0:
                    chapter_audios.append(self.read_passage(voice, text, **kwargs))

            all_chapter_audios.append(chapter_audios)

        return all_chapter_audios

    def extract_text(self, item):
        soup = BeautifulSoup(item.content, 'html.parser')
        return soup.body.get_text(separator=' ').strip().replace('\n', '.')