from monotonic_align import maximum_path
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
import random
import phonemizer

from nltk.tokenize import word_tokenize
from munch import Munch

from models import *
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule


def maximum_path(neg_cent, mask):
	""" Cython optimized version.
	neg_cent: [b, t_t, t_s]
	mask: [b, t_t, t_s]
	"""
	device = neg_cent.device
	dtype = neg_cent.dtype
	neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
	path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

	t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
	t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
	maximum_path_c(path, neg_cent, t_t_max, t_s_max)
	return torch.from_numpy(path).to(device=device, dtype=dtype)


def get_data_path_list(train_path=None, val_path=None):
	if train_path is None:
		train_path = "Data/train_list.txt"
	if val_path is None:
		val_path = "Data/val_list.txt"

	with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
		train_list = f.readlines()
	with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
		val_list = f.readlines()

	return train_list, val_list


def length_to_mask(lengths):
	mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
	mask = torch.gt(mask + 1, lengths.unsqueeze(1))
	return mask


# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
	"""
    normalized log mel -> mel -> norm -> log(norm)
    """
	x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
	return x


def get_image(arrs):
	plt.switch_backend('agg')
	fig = plt.figure()
	ax = plt.gca()
	ax.imshow(arrs)

	return fig

def recursive_munch(d):
    try:
        if isinstance(d, dict):
            return Munch((k, recursive_munch(v)) for k, v in d.items())
        elif isinstance(d, list):
            return [recursive_munch(v) for v in d]
        else:
            return d
    except Exception as e:
        print(f"Error in recursive_munch: {e}")
        raise


def log_print(message, logger):
    try:
        logger.info(message)
        print(message)
    except Exception as e:
        print(f"Error in log_print: {e}")
        raise


def set_seeds(seed=0):
    try:
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
    except Exception as e:
        print(f"Error in set_seeds: {e}")
        raise


def load_configurations(config_path):
    try:
        print(f"Loading configurations from: {config_path}")
        with open(config_path, 'r') as f:
            configurations = yaml.safe_load(f)
        print(f"Configurations loaded: {configurations}")
        return configurations
    except Exception as e:
        print(f"Error loading configurations: {e}")
        raise





def load_models(config, device):
    try:
        print("Loading ASR models")
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)
        print("ASR models loaded")

        print("Loading F0 models")
        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)
        print("F0 models loaded")

        print("Loading BERT models")
        BERT_path = config.get('PLBERT_dir', False)
        from Utils.PLBERT.util import load_plbert
        plbert = load_plbert(BERT_path)
        print("BERT models loaded")

        model_params = recursive_munch(config['model_params'])
        model = build_model(model_params, text_aligner, pitch_extractor, plbert)
        _ = [model[key].eval() for key in model]
        _ = [model[key].to(device) for key in model]

        print("Models loaded successfully")
        return model, model_params
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


def load_pretrained_model(model, model_path):
    try:
        print(f"Loading pretrained model from: {model_path}")
        params_whole = torch.load(model_path, map_location='cpu')
        params = params_whole['net']
        for key in model:
            if key in params:
                print(f'{key} loaded')
                try:
                    model[key].load_state_dict(params[key])
                except Exception as e:
                    print(f"Error loading state_dict for {key}: {e}")
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    model[key].load_state_dict(new_state_dict, strict=False)
        _ = [model[key].eval() for key in model]
        print("Pretrained model loaded successfully")
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
        raise


def create_sampler(model):
    try:
        print("Creating sampler")
        return DiffusionSampler(
            model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False
        )
    except Exception as e:
        print(f"Error creating sampler: {e}")
        raise


def length_to_mask(lengths):
    try:
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask
    except Exception as e:
        print(f"Error in length_to_mask: {e}")
        raise


def preprocess(wave, to_mel, mean, std):
    try:
        print("Preprocessing wave")
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor
    except Exception as e:
        print(f"Error in preprocess: {e}")
        raise


def compute_style(path, model, to_mel, mean, std, device):
    try:
        print(f"Computing style for path: {path}")
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio, to_mel, mean, std).to(device)
        with torch.no_grad():
            ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
        print("Style computed successfully")
        return torch.cat([ref_s, ref_p], dim=1)
    except Exception as e:
        print(f"Error computing style: {e}")
        raise


def load_phonemizer():
    try:
        print("Loading phonemizer")
        global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,
                                                             with_stress=True)
        print("Phonemizer loaded successfully")
        return global_phonemizer
    except Exception as e:
        print(f"Error loading phonemizer: {e}")
        raise


def inference(text, ref_s, model, sampler, textcleaner, to_mel, device, model_params, global_phonemizer, alpha=0.3,
              beta=0.7, diffusion_steps=5, embedding_scale=1):
    try:
        print(f"Running inference for text: {text}")
        text = text.strip()
        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
            text_mask = length_to_mask(input_lengths).to(device)

            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = sampler(noise=torch.randn((1, 256)).unsqueeze(1).to(device),
                             embedding=bert_dur,
                             embedding_scale=embedding_scale,
                             features=ref_s,
                             num_steps=diffusion_steps).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
            if model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new

            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
            if model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new

            out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        print("Inference completed successfully")
        return out.squeeze().cpu().numpy()[..., :-50]
    except Exception as e:
        print(f"Error during inference: {e}")
        raise


def get_voice_dir():
    try:
        target = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../voices')
        if not os.path.exists(target):
            target = os.path.dirname('./voices/')

        os.makedirs(target, exist_ok=True)

        return target
    except Exception as e:
        print(f"Error getting voice directory: {e}")
        raise

def get_voice(name, dir=get_voice_dir(), load_latents=True, extensions=["wav", "mp3", "flac"]):
    try:
        subj = os.path.join(dir, name)  # Use os.path.join for better path handling
        if not os.path.isdir(subj):
            print(f"No directory found for {name} in {dir}")
            return []

        # Optionally include 'pth' files
        if load_latents:
            extensions.append("pth")

        files = os.listdir(subj)
        voice = []
        for file in files:

            file_path = os.path.join(subj, file)
            if not os.path.isfile(file_path):  # Ensure the path points to a file
                continue

            ext = os.path.splitext(file)[-1][1:]  # Extract the extension without the dot
            if ext in extensions:
                voice.append(file_path)

        return sorted(voice)
    except Exception as e:
        print(f"Error getting voice: {e}")
        raise


def get_voice_list(dir=get_voice_dir(), append_defaults=False,
                   extensions=["wav", "mp3", "flac", "pth", "opus", "m4a", "webm", "mp4"]):
    try:
        defaults = []
        os.makedirs(dir, exist_ok=True)
        res = []

        for name in os.listdir(dir):
            if name in defaults:
                continue
            if not os.path.isdir(f'{dir}/{name}'):
                continue
            if len(os.listdir(os.path.join(dir, name))) == 0:
                continue
            files = get_voice(name, dir=dir, extensions=extensions)

            if len(files) > 0:
                res.append(name)
            else:
                for subdir in os.listdir(f'{dir}/{name}'):
                    if not os.path.isdir(f'{dir}/{name}/{subdir}'):
                        continue
                    files = get_voice(f'{name}/{subdir}', dir=dir, extensions=extensions)
                    if len(files) == 0:
                        continue
                    res.append(f'{name}/{subdir}')

        res = sorted(res)

        if append_defaults:
            res = res + defaults

        return res
    except Exception as e:
        print(f"Error getting voice list: {e}")
        raise


def load_models_webui(config,sigma_value, device="cpu", configuration_path="Models/model_paths.yml"):
    try:
        print(f"Loading models from configuration path: {configuration_path}")
        config = load_configurations()
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)
        print("ASR models loaded")

        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)
        print("F0 models loaded")

        BERT_path = config.get('PLBERT_dir', False)
        from Utils.PLBERT.util import load_plbert
        plbert = load_plbert(BERT_path)
        print("BERT models loaded")

        model_params = recursive_munch(config['model_params'])
        model_params.diffusion.dist.sigma_data = sigma_value
        model = build_model(model_params, text_aligner, pitch_extractor, plbert)

        _ = [model[key].eval() for key in model]
        _ = [model[key].to(device) for key in model]

        print("Models loaded successfully")
        return model, model_params
    except Exception as e:
        print(f"Error loading models from web UI: {e}")
        raise
