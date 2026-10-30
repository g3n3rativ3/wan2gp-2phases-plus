"""H3 Upscaler tab: run MiniMax H3's phase 2 on an existing video.

The tab queues a normal WanGP two-phase generation for a MiniMax H3 FL2VA model in which:
* phase 1 is replaced by the source video, encoded with the H3 VAE at the phase 1 resolution (see h3_phase2_patch);
* the phase 1 -> phase 2 scale is the chosen factor (x1.5 to x3, output limited to 4096 pixels) instead of WanGP's fixed x2;
* the source soundtrack is given as the FL2VA soundtrack ("Soundtrack Kept"), so the audio is not regenerated;
* phase 2 runs with the chosen noise level start, phase 2 Turbo LoRA multiplier and prompt.
"""

import math
import os
import shutil
import tempfile
import time

import gradio as gr

from . import h3_phase2_patch as patch

FL2VA_ARCHITECTURES = ("minimax_h3_fl2va_pruned", "minimax_h3_fl2va")
REF2VA_ARCHITECTURES = ("minimax_h3_ref2va_pruned", "minimax_h3_ref2va")
REFMOD_SETTING = "h3_refmod_state"  # custom setting of the MiniMax H3 RefMods plugin (inline panel selection)
REFERENCE_VIDEO_SLOTS = 3
REFERENCE_VIDEO_MODES = {1: "V-U", 2: "V+-U", 3: "V+*-U"}  # Ref2VA: one, two or three whole reference videos
DEFAULT_MODEL = "minimax_h3_fl2va_pruned"
FACTOR_CHOICES = [("x1 (refiner)", 1.0), ("x1.5", 1.5), ("x2", 2.0), ("x2.5", 2.5), ("x3", 3.0)]
MAX_OUTPUT_SIZE = 4096  # longest output side, in pixels
DEFAULT_FACTOR = 2.0
DEFAULT_NOISE = 0.6
DEFAULT_PHASE_2_STEPS = patch.PHASE_2_STEPS_DEFAULT  # WanGP 18+ "Phase 2 Steps" (older WanGP: fixed by the noise level start)
DEFAULT_MULTIPLIER = 0.5
DEFAULT_PROMPT = "."
PHASE_2_MODE_TURBO, PHASE_2_MODE_REGENERATE = "turbo", "regenerate"
PHASE_2_MODE_CHOICES = [("Turbo (WanGP default phase 2)", PHASE_2_MODE_TURBO), ("Open-H3 Regenerate 2K", PHASE_2_MODE_REGENERATE)]
# WanGP 18.8+ "H3 Regenerate 2K 2 Phases" preset: refiner LoRA (as an accelerator, phase 2 only) and its three-step phase 2 schedule
REGENERATE_LORA_URL = "https://huggingface.co/DeepBeepMeep/MiniMax-H3/resolve/main/loras/open_h3_regenerate_2k_lora.safetensors"
REGENERATE_SIGMAS = (0.909375, 0.8571428571428571, 0.6666666666666666)
FPS = 24  # MiniMax H3 default frame rate
FPS_CHOICES = [(f"Model Default ({FPS} fps)", ""), ("Source Video fps", "source")] + [(value, value) for value in ("15", "16", "23", "24", "25", "30", "48", "50")]
FRAMES_MIN, FRAMES_STEP, FRAMES_OFFSET = 107, 17, 5  # MiniMax H3: 107 frames minimum, then 5 + 17n
MAX_SECONDS = 600  # longest video processed (10 minutes), split into sliding windows: VRAM stays at the level of one window
RAM_BYTES_PER_PIXEL = 3  # WanGP keeps every output frame in RAM (uint8 RGB) until the video is saved
RAM_WARNING_SHARE = 0.6  # warn when the frames alone need more than this share of the system RAM (the models need RAM too)
WINDOW_MIN, WINDOW_MAX, WINDOW_STEP, WINDOW_DEFAULT = 124, 481, 17, 481  # H3 FL2VA sliding window sizes
OVERLAP_MIN, OVERLAP_MAX, OVERLAP_STEP, OVERLAP_DEFAULT = 1, 120, 17, 18
SIZE_ALIGNMENT = 32
SOURCE_HEIGHT = 200  # compact video inputs
REFERENCE_HEIGHT = 170
CONFIG_SLOTS = 4  # Text Encoder, Video VAE, DiT Denoising Priority, user configs
WORK_DIR = os.path.join(tempfile.gettempdir(), "wan2gp_h3_upscaler")

INTRO = ("Runs MiniMax H3's **phase 2** on an existing video: the video is encoded at its own resolution, enlarged by the "
         "H3 latent upscaler and refined by phase 2 at the target resolution. The soundtrack of the video is kept. "
         f"Videos are processed at 24 fps by default, from 107 frames (4.5 s at 24 fps) up to {MAX_SECONDS // 60} minutes, in sliding windows "
         "beyond the window size (VRAM stays at the level of one window; RAM grows with the length, see the estimate below the video), "
         f"and the output is limited to {MAX_OUTPUT_SIZE} pixels on its longest side.")


def _model_choices():
    wgp = patch._find_wgp_module()
    choices = []
    if wgp is not None:
        for model_type, model_def in wgp.models_def.items():
            if not isinstance(model_def, dict) or not model_def.get("visible", True):
                continue
            if str(model_def.get("architecture", "")) in FL2VA_ARCHITECTURES + REF2VA_ARCHITECTURES and model_def.get("phase_2_spatial_tiling", False):
                choices.append((str(model_def.get("name", model_type)), model_type))
    choices.sort(key=lambda choice: (choice[1] != DEFAULT_MODEL, choice[0].lower()))
    return choices


def _probe(path):
    from shared.utils.video_decode import probe_video_stream_metadata
    metadata = probe_video_stream_metadata(path)
    if not metadata:
        raise gr.Error("This file cannot be read as a video.")
    return metadata


def _frames_at_fps(metadata, target_fps):
    fps, frame_count, duration = float(metadata.get("fps_float") or 0.0), int(metadata.get("frame_count") or 0), float(metadata.get("duration") or 0.0)
    if fps > 0 and frame_count > 0:
        return int(math.floor(frame_count / fps * target_fps + 1e-6))
    return int(math.floor(duration * target_fps + 1e-6))


def _resolve_fps(fps_choice, metadata):
    """(generation fps, force_fps setting value)."""
    fps_choice = str(fps_choice or "")
    if fps_choice == "source":
        source_fps = max(1, int(round(float(metadata.get("fps_float") or FPS))))
        return source_fps, str(source_fps)
    if fps_choice.isdigit():
        return int(fps_choice), fps_choice
    return FPS, ""


def _frames_max(fps):
    """Longest H3 length (5 + 17n frames) within MAX_SECONDS at this frame rate."""
    frames = int(MAX_SECONDS * float(fps or FPS))
    return max(FRAMES_MIN, (frames - FRAMES_OFFSET) // FRAMES_STEP * FRAMES_STEP + FRAMES_OFFSET)


def _system_ram_bytes():
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        try:
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except Exception:
            return 0


def _ram_estimate_text(frames, width, height):
    """RAM used by the output frames, which WanGP keeps in memory until the video is saved."""
    needed = frames * width * height * RAM_BYTES_PER_PIXEL
    total = _system_ram_bytes()
    text = f"**RAM for the output frames:** about {needed / 1024 ** 3:.1f} GB"
    if total > 0:
        text += f" (system RAM: {total / 1024 ** 3:.0f} GB)"
        if needed > total * RAM_WARNING_SHARE:
            text = (f"⚠️ {text}: more than {int(RAM_WARNING_SHARE * 100)} % of the system RAM, which WanGP also needs for the models. "
                    "The upscale may fail: use a lower factor or a shorter video.")
    return text


def _usable_frames(frames_24, fps=FPS):
    frames = min(frames_24, _frames_max(fps))
    if frames < FRAMES_MIN:
        return 0
    return (frames - FRAMES_OFFSET) // FRAMES_STEP * FRAMES_STEP + FRAMES_OFFSET


def _aligned(value):
    return max(SIZE_ALIGNMENT, int(round(value / SIZE_ALIGNMENT)) * SIZE_ALIGNMENT)


def _target_size(width, height, factor):
    """Output size for the factor, with the longest side limited to MAX_OUTPUT_SIZE. Returns (width, height, effective factor)."""
    phase_1_width, phase_1_height = _aligned(width), _aligned(height)
    factor = max(1.0, min(float(factor), MAX_OUTPUT_SIZE / max(phase_1_width, phase_1_height)))
    target_width = min(MAX_OUTPUT_SIZE, _aligned(phase_1_width * factor))
    target_height = min(MAX_OUTPUT_SIZE, _aligned(phase_1_height * factor))
    effective = (target_width / phase_1_width + target_height / phase_1_height) / 2.0
    return target_width, target_height, effective


def _plan(path, factor, fps_choice=""):
    metadata = _probe(path)
    width, height = int(metadata.get("display_width") or metadata.get("width")), int(metadata.get("display_height") or metadata.get("height"))
    fps, force_fps = _resolve_fps(fps_choice, metadata)
    frames_available = _frames_at_fps(metadata, fps)
    frames = _usable_frames(frames_available, fps)
    if max(_aligned(width), _aligned(height)) > MAX_OUTPUT_SIZE:
        raise gr.Error(f"The video is larger than {MAX_OUTPUT_SIZE} pixels: it cannot be processed.")
    target_width, target_height, effective_factor = _target_size(width, height, factor)
    return {"width": width, "height": height, "source_fps": float(metadata.get("fps_float") or 0.0), "fps": fps, "force_fps": force_fps,
            "frames_available": frames_available, "frames": frames, "target_width": target_width, "target_height": target_height,
            "factor": float(factor), "effective_factor": effective_factor}


def _window_count(frames, window_size, overlap):
    if frames <= window_size:
        return 1
    return 1 + math.ceil((frames - window_size) / max(1, window_size - overlap))


def _plan_text(path, factor, fps_choice="", window_size=WINDOW_DEFAULT, overlap=OVERLAP_DEFAULT):
    if not path:
        return ""
    try:
        plan = _plan(path, float(factor), fps_choice)
    except gr.Error as error:
        return f"⚠️ {error}"
    fps = plan["fps"]
    lines = [f"**Source:** {plan['width']}x{plan['height']}, {plan['source_fps']:.3g} fps, {plan['frames_available']} frames at {fps} fps "
             f"({plan['frames_available'] / fps:.2f} s)",
             f"**Output:** {plan['target_width']}x{plan['target_height']} at {fps} fps" + (" (refiner: same size, no upscale)" if plan["factor"] <= 1.0 else "")]
    if plan["effective_factor"] < plan["factor"] - 0.01:
        lines.append(f"The output is limited to {MAX_OUTPUT_SIZE} pixels: effective factor x{plan['effective_factor']:.2f} instead of x{plan['factor']:g}.")
    if plan["frames"] == 0:
        lines.append(f"⚠️ The video is too short: MiniMax H3 needs at least {FRAMES_MIN} frames ({FRAMES_MIN / fps:.2f} s at {fps} fps).")
    else:
        lines[1] += f", {plan['frames']} frames ({plan['frames'] / fps:.2f} s)"
        windows = _window_count(plan["frames"], int(window_size or WINDOW_DEFAULT), int(overlap or OVERLAP_DEFAULT))
        lines[1] += ", 1 window" if windows == 1 else f", about {windows} sliding windows of {int(window_size)} frames (overlap {int(overlap)})"
        if plan["frames"] < plan["frames_available"]:
            dropped = plan["frames_available"] - plan["frames"]
            frames_max = _frames_max(fps)
            reason = (f"the maximum is {frames_max} frames ({MAX_SECONDS // 60} minutes)" if plan["frames_available"] > frames_max
                      else "H3 lengths are 5 + 17n frames")
            lines.append(f"The last {dropped} frame(s) ({dropped / fps:.2f} s) are not processed: {reason}.")
    if plan["frames"] > 0:
        lines.append(_ram_estimate_text(plan["frames"], plan["target_width"], plan["target_height"]))
    return "<br>".join(lines)


SOUNDTRACK_MARGIN_SECONDS = 3.0  # soundtrack kept beyond the processed duration (see _prepare_files)


def _prepare_files(path, frames, fps=FPS):
    """Copy the source into a work folder and write its soundtrack as a WAV covering the processed duration plus a margin
    (silence where the video has no more audio).

    The margin matters with sliding windows: WanGP's last window can be a few frames longer than the video (H3 window
    lengths are 5 + 17n frames, at least 107), and H3 also generates a short audio tail after each window. Where the
    soundtrack does not cover a window, WanGP keeps the audio generated for that window instead of the soundtrack, and in
    this tab phase 1 is not denoised (it is replaced by the source), so that audio would be pure noise. WanGP trims the
    final soundtrack to the video duration, so the margin never reaches the output."""
    import numpy as np
    import soundfile as sf
    from shared.utils.audio_video import extract_audio_track_to_wav

    os.makedirs(WORK_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    source_copy = os.path.join(WORK_DIR, f"{stamp}_source{os.path.splitext(path)[1] or '.mp4'}")
    shutil.copyfile(path, source_copy)
    wav_path = os.path.join(WORK_DIR, f"{stamp}_soundtrack.wav")
    duration = frames / fps + SOUNDTRACK_MARGIN_SECONDS
    samples, sample_rate = None, 32000
    try:
        if extract_audio_track_to_wav(source_copy, wav_path):
            samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    except Exception as error:
        print(f"{patch.LOG_PREFIX} H3 Upscaler: no usable soundtrack ({error}), silence is used")
        samples = None
    if samples is None or samples.shape[0] == 0:
        samples, sample_rate = np.zeros((1, 2), dtype=np.float32), 32000
        print(f"{patch.LOG_PREFIX} H3 Upscaler: the video has no soundtrack, silence is used")
    needed = int(round(duration * sample_rate))
    if samples.shape[0] < needed:
        samples = np.concatenate((samples, np.zeros((needed - samples.shape[0], samples.shape[1]), dtype=samples.dtype)))
    sf.write(wav_path, samples[:needed], sample_rate)
    return source_copy, wav_path


def _wgp():
    return patch._find_wgp_module()


def _config_groups(model_type):
    """[(label, choices, visible)] for the config groups (Text Encoder, Video VAE, DiT Denoising Priority, user configs) of a model,
    as in Misc., padded to CONFIG_SLOTS."""
    wgp = _wgp()
    groups = []
    if wgp is not None and model_type:
        try:
            from shared import config_groups as model_config_groups
            model_def = wgp.get_model_def(model_type)
            for configs in wgp.get_model_config_groups(model_type, model_def):
                items = model_config_groups.get_config_items(configs)
                choices = [(model_config_groups.get_default_label(configs), "")] + [(config_def.get("name", config_id), config_id) for config_id, config_def in items]
                groups.append((model_config_groups.get_config_name(configs), choices, bool(items)))
        except Exception as error:
            print(f"{patch.LOG_PREFIX} H3 Upscaler: model configs unavailable ({error})")
            groups = []
    groups = groups[:CONFIG_SLOTS]
    return groups + [(f"Config {index + 1}", [("Default", "")], False) for index in range(len(groups), CONFIG_SLOTS)]


def _config_dropdown_updates(model_type):
    return [gr.update(choices=choices, value="", label=label, visible=visible) for label, choices, visible in _config_groups(model_type)]


def _profile_choices():
    wgp = _wgp()
    return [("Default Video Memory Profile", -1)] + list(getattr(wgp, "memory_profile_choices", []) or [])


def _attention_choices(model_type):
    wgp = _wgp()
    model_def = (wgp.get_model_def(model_type) if wgp is not None and model_type else None) or {}
    choices = [("Default Attention Mode", "")]
    if model_def.get("default_attention_modes_supported", True):
        choices += list(getattr(wgp, "attention_modes_choices", []) or [])
    for mode, definition in (model_def.get("custom_attention_modes") or {}).items():
        choices.append((f"{mode}: {definition.get('label', mode)}", mode))
    return choices


def _is_ref2va(model_type):
    wgp = _wgp()
    model_def = (wgp.get_model_def(model_type) if wgp is not None and model_type else None) or {}
    return str(model_def.get("architecture", "")) in REF2VA_ARCHITECTURES


REFMOD_PLUGIN_ID = "H3RefMods"  # PlugIn_Id of the MiniMax H3 RefMods plugin
REFMOD_PLUGIN_CLASS = "MiniMaxH3RefModsPlugin"


def _enabled_plugin_module(matches):
    """The plugin module (its plugin.py) of an enabled plugin for which matches(module) is true, else None.
    WanGP only imports the plugin.py of enabled plugins; the enabled list of the WanGP config is checked too."""
    import sys
    enabled = None
    wgp = _wgp()
    if wgp is not None and isinstance(getattr(wgp, "server_config", None), dict):
        enabled = set(wgp.server_config.get("enabled_plugins") or [])
    for name, module in list(sys.modules.items()):
        try:
            if module is None or not matches(module):
                continue
        except Exception:
            continue
        package = name.rsplit(".", 1)[0] if "." in name else name
        if enabled is not None and package not in enabled and name not in enabled:
            continue
        return module
    return None


def _refmod_plugin_module():
    """The plugin module of the MiniMax H3 RefMods plugin when it is enabled, else None."""
    return _enabled_plugin_module(lambda module: getattr(module, "PlugIn_Id", None) == REFMOD_PLUGIN_ID and hasattr(module, REFMOD_PLUGIN_CLASS))


MULTIGPU_PLUGIN_CLASS = "MultiGPUPlugin"
MULTIGPU_PLUGIN_NAME = "Multi-GPU"
# Labels of the Multi-GPU tab controls -> keys of that plugin's saved settings (used to show the current values when the tab is opened)
MULTIGPU_LABEL_KEYS = {
    "Enable multi-GPU": "enabled",
    "Split H3 denoising across cuda:0 and cuda:1": "denoise_enabled",
    "Denoising work share of cuda:0 (%)": "denoise_share_cuda0",
    "Also split still images (Text to Image)": "denoise_images",
    "Attention heads per exchanged chunk": "denoise_chunk_heads",
    "Split H3 VAE decoding across cuda:0 and cuda:1": "vae_enabled",
    "Work share of cuda:0 (%)": "share_cuda0",
    "Keep the last text encoder layers on cuda:1 (ignored while denoising is split)": "te_enabled",
    "Text encoder layers kept on cuda:1": "te_layers_cuda1",
}


def _multigpu_plugin_module():
    """The plugin module of the Multi-GPU plugin when it is enabled, else None."""
    return _enabled_plugin_module(lambda module: getattr(module, "PLUGIN_NAME", None) == MULTIGPU_PLUGIN_NAME
                                  and callable(getattr(getattr(module, MULTIGPU_PLUGIN_CLASS, None), "_build_tab", None)))


def _multigpu_settings(multigpu_module):
    """The settings module of the Multi-GPU plugin (shared by its Media Generator tab and the copy in this tab), or None."""
    import sys
    package = multigpu_module.__name__.rsplit(".", 1)[0]
    settings = sys.modules.get(f"{package}.settings")
    return settings if callable(getattr(settings, "get", None)) else None


LATENT_PLUGIN_CLASS = "H3LatentPlugin"
LATENT_TASK_KEY = "_h3_latent_task_options_v1"  # H3 Latent Continue: per-task options (moved into the task's plugin_data by that plugin)
LATENT_SAVE_LABEL = "Save latent checkpoint next to output video"


def _latent_plugin_module():
    """The plugin module of the H3 Latent Continue plugin when it is enabled, else None."""
    return _enabled_plugin_module(lambda module: hasattr(module, LATENT_PLUGIN_CLASS) and callable(getattr(module, "make_generation_wrapper", None)))


def _latent_task_key(latent_module):
    import sys
    package = latent_module.__name__.rsplit(".", 1)[0]
    integration = sys.modules.get(f"{package}.integration")
    return getattr(integration, "TASK_KEY", LATENT_TASK_KEY)


def _allow_upscale_soundtrack(latent_module):
    """H3 Latent Continue rejects the Ref2VA 'kept soundtrack' audio mode ("S"), but accepts FL2VA's "A", which the H3
    pipeline handles exactly the same way (the target audio is conditioned on the encoded soundtrack). Ref2VA upscales
    need "AS" ("A" alone is a voice reference in Ref2VA), so the check is relaxed for H3 Upscaler tasks only."""
    import sys
    package = latent_module.__name__.rsplit(".", 1)[0]
    integration = sys.modules.get(f"{package}.integration")
    original = getattr(integration, "preflight", None)
    if not callable(original) or getattr(original, "_two_phases_plus_upscale", False):
        return

    def preflight(params, opts):
        custom_settings = params.get("custom_settings") if isinstance(params, dict) else None
        audio_prompt_type = str(params.get("audio_prompt_type") or "") if isinstance(params, dict) else ""
        if isinstance(custom_settings, dict) and custom_settings.get(patch.UPSCALE_SOURCE_KEY) and "S" in audio_prompt_type:
            params = dict(params, audio_prompt_type=audio_prompt_type.replace("S", ""))  # same pipeline path as FL2VA "A"
        return original(params, opts)

    preflight._two_phases_plus_upscale = True
    preflight.__wrapped__ = original
    integration.preflight = preflight


def _latent_save_problem(ref2va, tiling, frames, window_size):
    """Why H3 Latent Continue cannot save a checkpoint for this upscale (its own limits), or None."""
    if tiling:
        return "H3 Latent Continue does not support Phase 2 with Tiling: uncheck it in Advanced > Misc."
    if frames > int(window_size):
        return (f"H3 Latent Continue saves one generated window per job: this video needs {frames} frames, more than the sliding window "
                f"size ({int(window_size)}). Shorten the video or raise Sliding Window Size (maximum {WINDOW_MAX}).")
    return None


def _regenerate_supported():
    """WanGP 18.8+: Custom Sigmas and the Open-H3 Regenerate phase 2 (silent phase 2 audio, phase 1 audio kept)."""
    wgp = _wgp()
    if wgp is None or not callable(getattr(wgp, "parse_custom_sigmas", None)):
        return False
    try:
        from models.minimax_h3 import constants
    except Exception:
        return False
    return bool(getattr(constants, "H3_REGENERATE_LORA", None))


def _regenerate_loras(loras, loras_multipliers):
    """The Regenerate LoRA as the only accelerator (phase 2 only), followed by the additional LoRAs of Media Generator,
    as WanGP's preset does (accelerator LoRAs, the part before '|', are replaced)."""
    loras, loras_multipliers = list(loras or []), loras_multipliers or ""
    try:
        from shared.utils.loras_mutipliers import extract_loras_side
        loras, loras_multipliers = extract_loras_side(loras, loras_multipliers, "after")
    except Exception:
        pass
    return [REGENERATE_LORA_URL] + loras, "0;1|" + (loras_multipliers or "").strip()


def _phase_2_info(mode, noise_value, steps_value):
    if mode == PHASE_2_MODE_REGENERATE:
        return (f"**Phase 2: {len(REGENERATE_SIGMAS)} steps** (Open-H3 Regenerate 2K: sigmas {', '.join(f'{value:.4g}' for value in REGENERATE_SIGMAS)}, 0). "
                "Refiner LoRA instead of the Turbo LoRA, made for high-resolution detail; the soundtrack is kept.")
    return patch.phase_2_steps_text(noise_value, steps_value)


def _component_tree(block):
    for child in getattr(block, "children", None) or []:
        yield child
        yield from _component_tree(child)


def _build_multigpu_tab(multigpu_module):
    """Add the Multi-GPU plugin's own tab (same controls, same saved settings) and refresh its values each time it is
    opened, since the same settings can also be changed from the Media Generator tab."""
    try:
        tab = getattr(multigpu_module, MULTIGPU_PLUGIN_CLASS)._build_tab(None)
    except Exception as error:
        print(f"{patch.LOG_PREFIX} H3 Upscaler: could not build the Multi-GPU tab: {error!r}")
        return None
    settings = _multigpu_settings(multigpu_module)
    controls = [(component, MULTIGPU_LABEL_KEYS[component.label]) for component in _component_tree(tab)
                if isinstance(getattr(component, "label", None), str) and component.label in MULTIGPU_LABEL_KEYS]
    if settings is not None and controls and hasattr(tab, "select"):
        def current_values():
            values = settings.get()
            return [gr.update(value=values[key]) if key in values else gr.update() for _, key in controls]
        tab.select(fn=current_values, inputs=None, outputs=[component for component, _ in controls], show_progress="hidden", queue=False)
    return tab


def _refmod_payload(refmod_module, values):
    """RefMods selection (JSON) from the (mod, strength) values of the inline picker rows, exactly as the RefMods plugin builds it."""
    import json
    none_choice = getattr(refmod_module, "NONE_CHOICE", "(none)")
    rows = []
    for index in range(0, len(values) - 1, 2):
        mod, strength = values[index], values[index + 1]
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            continue
        if mod and mod != none_choice and strength > 0:
            rows.append({"mod": mod, "strength": strength})
    if not rows:
        return None, 0
    payload = {"rows": rows, "retention": 1.0, "scramble_seed": -1, "curve": None}
    if len(values) % 2 == 1:
        # RefMods 0.36+: the panel ends with the "Video mods on sliding windows 2, 3, ..." dropdown (a one-element row)
        payload["window_mode"] = values[-1] or getattr(refmod_module, "WINDOW_MODE_DEFAULT", "still")
    return json.dumps(payload), len(rows)


def _refmods_status(refmod_module, *values):
    _, count = _refmod_payload(refmod_module, values)
    return f"✅ {count} RefMod(s) armed for the next Ref2VA upscale from this tab." if count else "*No RefMods selected.*"


def _media_generator_loras(state, model_type):
    """(activated_loras, loras_multipliers) selected in Media Generator for this model (its saved form settings)."""
    wgp = _wgp()
    if wgp is None or not isinstance(state, dict) or not model_type:
        return [], ""
    try:
        settings = wgp.get_model_settings(state, model_type) or {}
    except Exception:
        settings = {}
    loras = list(settings.get("activated_loras") or [])
    multipliers = settings.get("loras_multipliers") or ""
    return loras, multipliers if isinstance(multipliers, str) else " ".join(str(value) for value in multipliers)


def _phase_2_multiplier(multiplier):
    """The multiplier phase 2 uses: 'a;b' -> b, 'a' -> a (comma lists are per-step values)."""
    return str(multiplier).split(";")[-1] if multiplier is not None else "1"


def _loras_text(state, model_type):
    if not model_type:
        return ""
    loras, multipliers = _media_generator_loras(state, model_type)
    if not loras:
        return f"No LoRA is selected in **Media Generator** for this model (`{model_type}`)."
    try:
        from shared.utils.loras_mutipliers import preparse_loras_multipliers
        values = preparse_loras_multipliers(multipliers) if multipliers else []
    except Exception:
        values = str(multipliers).split()
    lines = [f"LoRAs selected in **Media Generator** for `{model_type}`:"]
    for index, lora in enumerate(loras):
        value = values[index] if index < len(values) else "1"
        name = os.path.basename(str(lora).split("|", 1)[0])
        phase_2 = "0 (Turbo LoRAs are disabled in phase 2)" if "turbo" in name.lower() else _phase_2_multiplier(value)
        lines.append(f"- {name}: multiplier `{value}`, phase 2 uses **{phase_2}**")
    return "<br>".join(lines)


def _copy_inputs(paths, prefix):
    os.makedirs(WORK_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    copies = []
    for index, path in enumerate(paths):
        if not path or not os.path.isfile(path):
            continue
        target = os.path.join(WORK_DIR, f"{stamp}_{prefix}{index + 1}{os.path.splitext(path)[1] or '.bin'}")
        shutil.copyfile(path, target)
        copies.append(target)
    return copies


def _read_prompt(path):
    if not path:
        raise gr.Error("Load a video first.")
    try:
        from shared.utils.video_metadata import read_metadata_from_video
        metadata = read_metadata_from_video(path)
    except Exception:
        metadata = None
    prompt = metadata.get("prompt") if isinstance(metadata, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        gr.Info("This video has no prompt in its properties.")
        return gr.update()
    gr.Info("Prompt copied from the video properties.")
    return prompt


def build_settings(source_copy, wav_path, plan, factor, model_type, noise_level, multiplier, prompt, seed,
                   config="", override_profile=-1, override_attention="", tiling=False, loras=None, loras_multipliers="",
                   window_size=WINDOW_DEFAULT, overlap=OVERLAP_DEFAULT, ref2va=False, image_refs=None, reference_videos=None,
                   refmods=None, phase_2_steps=None, phase_2_mode=PHASE_2_MODE_TURBO):
    frames = plan["frames"]
    image_refs = list(image_refs or [])
    reference_videos = list(reference_videos or [])[:REFERENCE_VIDEO_SLOTS]
    video_prompt_type = ""
    if ref2va:
        video_prompt_type += "I" if image_refs else ""
        video_prompt_type += REFERENCE_VIDEO_MODES.get(len(reference_videos), "")
    video_prompt_type += "~" if tiling else ""  # "~" = Two Phases with Tiling
    settings = {
        "model_type": model_type,
        "prompt": prompt.strip() if isinstance(prompt, str) and prompt.strip() else DEFAULT_PROMPT,
        "resolution": f"{plan['target_width']}x{plan['target_height']}",
        "video_length": frames,
        "sliding_window_size": int(window_size),
        "sliding_window_overlap": int(overlap),
        "seed": int(seed) if seed is not None else -1,
        "num_inference_steps": 1,  # phase 1 is replaced by the source video: one step from pure noise to the source latent
        "guidance_phases": 2,
        "switch_threshold": float(noise_level),
        "guidance_scale": 1.0,
        "sample_solver": "euler",
        "skip_steps_cache_type": "",
        "image_prompt_type": "",
        "video_prompt_type": video_prompt_type,
        # Generate video based on the soundtrack (soundtrack kept): "A" for FL2VA, "AS" for Ref2VA (where "A" is a voice reference)
        "audio_prompt_type": "AS" if ref2va else "A",
        "audio_guide": wav_path,
        "activated_loras": list(loras or []),
        "loras_multipliers": loras_multipliers or "",
        "prompt_enhancer": "",
        "spatial_upsampling": "",
        "temporal_upsampling": "",
        "repeat_generation": 1,
        "force_fps": plan["force_fps"],
        "config": config or "",
        "override_profile": int(override_profile) if override_profile not in (None, "") else -1,
        "override_attention": override_attention or "",
        "custom_settings": {
            "audio_refinement": "none",
            patch.MULTIPLIER_KEY: float(multiplier),
            patch.PROMPT_KEY: "",
            patch.UPSCALE_SOURCE_KEY: source_copy,
            patch.UPSCALE_FACTOR_KEY: float(plan["effective_factor"]),
        },
    }
    if phase_2_steps is not None:
        settings["num_inference_steps2"] = int(phase_2_steps)
    wgp = _wgp()
    if wgp is not None and callable(getattr(wgp, "parse_custom_sigmas", None)):
        settings["custom_sigmas"] = ""  # WanGP 18.8+: the tab's own phase 2 schedule (Noise Level Start and Phase 2 Steps)
    if phase_2_mode == PHASE_2_MODE_REGENERATE:
        regenerate_loras, regenerate_multipliers = _regenerate_loras(settings["activated_loras"], settings["loras_multipliers"])
        settings.update({"activated_loras": regenerate_loras, "loras_multipliers": regenerate_multipliers,
                         "custom_sigmas": "; " + ", ".join(repr(value) for value in REGENERATE_SIGMAS),
                         "switch_threshold": REGENERATE_SIGMAS[0], "num_inference_steps2": len(REGENERATE_SIGMAS)})
    if ref2va:
        settings.update({"image_refs": image_refs or None, "image_refs_relative_size": 100, "remove_background_images_ref": 0})
        for index, key in enumerate(("video_guide", "video_guide2", "video_guide3")):
            settings[key] = reference_videos[index] if index < len(reference_videos) else None
        if refmods:
            settings["custom_settings"][REFMOD_SETTING] = refmods
    return settings


def create_tab(plugin, api_session):
    from shared import config_groups as model_config_groups
    from shared.gradio.gallery import AdvancedMediaGallery, get_gradio_file_path

    active_job = {"job": None}
    refmod_module = _refmod_plugin_module()
    multigpu_module = _multigpu_plugin_module()
    latent_module = _latent_plugin_module()
    model_choices = _model_choices()
    default_model = model_choices[0][1] if model_choices else None
    state_component = getattr(plugin, "state", None)
    state_input = [state_component] if state_component is not None else [gr.State(None)]

    with gr.Column():
        status = gr.Markdown("", elem_id="h3_upscaler_status")  # progress / result line, at the very top of the tab
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                source_video = gr.Video(label="Video to upscale", sources=["upload"], height=SOURCE_HEIGHT, elem_id="h3_upscaler_source")
            with gr.Column(scale=2):
                plan_info = gr.Markdown("")
                factor = gr.Radio(choices=FACTOR_CHOICES, value=DEFAULT_FACTOR, label=f"Upscale Factor (width and height, output limited to {MAX_OUTPUT_SIZE} pixels)")
                model = gr.Dropdown(choices=model_choices, value=default_model, label="MiniMax H3 Model (FL2VA or Ref2VA)",
                                    info="Model used for phase 2. Ref2VA can use reference images / videos" + (" and RefMods (Advanced > LoRAs)" if refmod_module is not None else "")
                                         + "; Turbo finetunes double the Turbo effect.")
        with gr.Column(visible=_is_ref2va(default_model)) as references:
            gr.Markdown("**References (Ref2VA)**: used by phase 2 to keep identities and appearance. Name them in the Phase 2 Prompt "
                        "(`<Picture 1>`, `<Video 1>`...) if the prompt describes them.")
            with gr.Row():
                with gr.Column(scale=2, min_width=260):
                    image_refs_gallery = AdvancedMediaGallery(media_mode="image", height=REFERENCE_HEIGHT, columns=4, label="Reference Images")
                    image_refs_gallery.mount()
                reference_videos = [gr.Video(label=f"Reference Video {index + 1}", sources=["upload"], height=REFERENCE_HEIGHT, min_width=160)
                                    for index in range(REFERENCE_VIDEO_SLOTS)]
        regenerate_supported = _regenerate_supported()
        phase_2_mode = gr.Radio(choices=PHASE_2_MODE_CHOICES, value=PHASE_2_MODE_TURBO, label="Phase 2 Mode", visible=regenerate_supported,
                                info="Open-H3 Regenerate 2K (WanGP 18.8+): a refiner LoRA with its own three-step schedule, for finer detail at "
                                     "high resolution. It replaces the Turbo LoRA, the noise level and the steps below; the soundtrack is kept.")
        with gr.Column() as turbo_controls:
            noise_level = gr.Slider(minimum=patch.NOISE_LEVEL_MIN, maximum=1.0, step=0.01, value=DEFAULT_NOISE,
                                    label=patch.noise_level_label(), show_reset_button=False)
            phase_2_steps = gr.Slider(minimum=1, maximum=10, step=1, value=DEFAULT_PHASE_2_STEPS, label="Phase 2 Steps",
                                      visible=patch.phase_2_steps_supported(), show_reset_button=False)
        steps_info = gr.Markdown(patch.phase_2_steps_text(DEFAULT_NOISE, DEFAULT_PHASE_2_STEPS))
        with gr.Column() as multiplier_controls:
            multiplier = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=DEFAULT_MULTIPLIER,
                                   label="Phase 2 Turbo LoRA Multiplier", show_reset_button=False)
        prompt = gr.Textbox(value=DEFAULT_PROMPT, lines=3, max_lines=12, label="Phase 2 Prompt",
                            placeholder="'.' = no description, phase 2 only refines what it sees.")
        with gr.Row():
            copy_prompt = gr.Button("Copy the Prompt from the Video Properties", size="sm")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)", min_width=160)
        with gr.Accordion("Advanced", open=False):
            with gr.Tabs():
                with gr.Tab("LoRAs"):
                    use_loras = gr.Checkbox(value=False, label="Apply the LoRAs selected in Media Generator for this model")
                    gr.Markdown("LoRAs are chosen in the **LoRAs** tab of **Media Generator** (with everything other plugins add there), "
                                "for the same model as above. Phase 1 is replaced by your video, so only the **phase 2** multiplier of each LoRA "
                                "matters: `a;b` uses `b`, a single value is used as is. Turbo LoRAs are set to 0 in phase 2, as in normal generations.")
                    loras_info = gr.Markdown("")
                    refresh_loras = gr.Button("Refresh from Media Generator", size="sm")
                    refmod_rows, refmods_section = [], None
                    if refmod_module is not None:
                        with gr.Accordion("MiniMax H3 RefMods (inline)", open=False, visible=_is_ref2va(default_model)) as refmods_section:
                            gr.Markdown("Applies only when a **MiniMax H3 Ref2VA** model is selected above. Same panel as in Media Generator, "
                                        "with its own selection: the RefMods picked here are used by this tab only. Extract new RefMods "
                                        "from the **MiniMax H3 RefMods** tab.")
                            try:
                                refmod_rows = list(getattr(refmod_module, REFMOD_PLUGIN_CLASS)._build_mod_picker_rows(None))
                            except Exception as error:
                                refmod_rows = []
                                print(f"{patch.LOG_PREFIX} H3 Upscaler: could not build the RefMods panel: {error!r}")
                                gr.Markdown(f"⚠️ The RefMods panel could not be built ({error!r}).")
                            refmods_info = gr.Markdown("*No RefMods selected.*")
                with gr.Tab("Sliding Window"):
                    window_size = gr.Slider(minimum=WINDOW_MIN, maximum=WINDOW_MAX, step=WINDOW_STEP, value=WINDOW_DEFAULT,
                                            label="Sliding Window Size", show_reset_button=False)
                    window_overlap = gr.Slider(minimum=OVERLAP_MIN, maximum=OVERLAP_MAX, step=OVERLAP_STEP, value=OVERLAP_DEFAULT,
                                               label="Windows Frames Overlap (needed to maintain continuity between windows)", show_reset_button=False)
                    gr.Markdown("Videos longer than the window size are processed in several windows. Each new window starts from the last "
                                "frames of the previous upscaled window, and its own part of the source video replaces its phase 1.")
                with gr.Tab("Misc."):
                    gr.Markdown("<B>Model config variations: they may impact RAM/VRAM consumption or performance</B>")
                    with gr.Row():
                        config_dropdowns = [gr.Dropdown(choices=choices, value="", visible=visible, label=label)
                                            for label, choices, visible in _config_groups(default_model)]
                    fps_choice = gr.Dropdown(choices=FPS_CHOICES, value="", label=f"Override Frames Per Second (model default={FPS} fps)")
                    override_profile = gr.Dropdown(choices=_profile_choices(), value=-1, label="Override Memory Profile")
                    override_attention = gr.Dropdown(choices=_attention_choices(default_model), value="", label="Override Attention Mode")
                    tiling = gr.Checkbox(value=False, label="Phase 2 with Tiling (lower peak VRAM for large outputs, possible seams)")
                if multigpu_module is not None:  # Multi-GPU plugin enabled: its own tab, sharing its settings with Media Generator
                    _build_multigpu_tab(multigpu_module)
        save_latent = gr.Checkbox(value=False, label=LATENT_SAVE_LABEL, visible=latent_module is not None,
                                  info="H3 Latent Continue plugin: saves the latents of the upscaled video (.safetensors, same name as the video) "
                                       "so it can be continued with latents in Media Generator. One window, untiled phase 2.")
        with gr.Row():
            start = gr.Button("Upscale", variant="primary")
            abort = gr.Button("Abort")

    def run_upscale(video_path, factor_value, model_type, noise_value, multiplier_value, prompt_value, seed_value,
                    fps_value, profile_value, attention_value, tiling_value, use_loras_value, window_value, overlap_value,
                    image_refs_value, phase_2_steps_value, save_latent_value, phase_2_mode_value, state_value, *rest):
        # A generator, run directly by WanGP's WebUI session wrapper (it keeps the Gradio session needed to queue the task).
        config_values = rest[:CONFIG_SLOTS]
        reference_video_values = rest[CONFIG_SLOTS:CONFIG_SLOTS + REFERENCE_VIDEO_SLOTS]
        refmod_values = rest[CONFIG_SLOTS + REFERENCE_VIDEO_SLOTS:]
        if not video_path:
            raise gr.Error("Load a video first.")
        if not model_type:
            raise gr.Error("No MiniMax H3 FL2VA or Ref2VA model is available.")
        ref2va = _is_ref2va(model_type)
        factor_value = float(factor_value)
        plan = _plan(video_path, factor_value, fps_value)
        if plan["frames"] == 0:
            raise gr.Error(f"The video is too short: MiniMax H3 needs at least {FRAMES_MIN} frames ({FRAMES_MIN / plan['fps']:.2f} s at {plan['fps']} fps).")
        save_latent_value = bool(save_latent_value) and latent_module is not None
        if save_latent_value:
            problem = _latent_save_problem(ref2va, bool(tiling_value), plan["frames"], window_value)
            if problem:
                raise gr.Error(problem)
        plugin.ensure_hooks()
        patch.install(verbose=False)
        yield "Preparing the source video and its soundtrack..."
        source_copy, wav_path = _prepare_files(video_path, plan["frames"], plan["fps"])
        config = model_config_groups.serialize_config_selection(*config_values)
        loras, loras_multipliers = _media_generator_loras(state_value, model_type) if use_loras_value else ([], "")
        image_refs, videos, refmods = [], [], None
        if ref2va:
            image_refs = _copy_inputs([get_gradio_file_path(item) for item in (image_refs_value or [])], "ref_image")
            videos = _copy_inputs([video for video in reference_video_values if video], "ref_video")
            refmods = _refmod_payload(refmod_module, refmod_values)[0] if refmod_module is not None and refmod_values else None
        settings = build_settings(source_copy, wav_path, plan, factor_value, model_type, noise_value, multiplier_value, prompt_value, seed_value,
                                  config=config, override_profile=profile_value, override_attention=attention_value, tiling=bool(tiling_value),
                                  loras=loras, loras_multipliers=loras_multipliers, window_size=window_value, overlap=overlap_value,
                                  ref2va=ref2va, image_refs=image_refs, reference_videos=videos, refmods=refmods,
                                  phase_2_steps=phase_2_steps_value if patch.phase_2_steps_supported() else None,
                                  phase_2_mode=phase_2_mode_value if regenerate_supported else PHASE_2_MODE_TURBO)
        if save_latent_value:
            if ref2va:
                _allow_upscale_soundtrack(latent_module)
            settings[_latent_task_key(latent_module)] = {"model_type": model_type,
                                                         "options": {"save": True, "continue": False, "latent_path": "", "join_mode": "baseline"}}
            settings.update({"film_grain_intensity": 0, "sliding_window_discard_last_frames": 0, "sliding_window_trim_first_frames": 0,
                             "sliding_window_color_correction_strength": 0})
        print(f"{patch.LOG_PREFIX} H3 Upscaler: {plan['width']}x{plan['height']} -> {settings['resolution']} (x{plan['effective_factor']:.2f}), "
              f"{plan['frames']} frames at {plan['fps']} fps, noise level start {float(noise_value):g}, Turbo multiplier {float(multiplier_value):g}, "
              f"model {model_type}{', tiled phase 2' if tiling_value else ''}{', config ' + config if config else ''}"
              f", sliding window {int(window_value)}/{int(overlap_value)}{', LoRAs ' + str(len(loras)) if loras else ''}"
              + (f", {len(image_refs)} reference image(s), {len(videos)} reference video(s){', RefMods' if refmods else ''}" if ref2va else "")
              + (", latent checkpoint saved (H3 Latent Continue)" if save_latent_value else "")
              + (", phase 2 Open-H3 Regenerate 2K" if regenerate_supported and phase_2_mode_value == PHASE_2_MODE_REGENERATE else ""))

        progress_state = {"text": "Queued in WanGP...", "changed": True}

        class Callbacks:
            def on_status(self, status_text):
                status_text = str(status_text or "").strip()
                if status_text:
                    progress_state.update(text=status_text, changed=True)

            def on_progress(self, update):
                percent = getattr(update, "progress", None)
                text = str(getattr(update, "status", "") or "Upscaling...")
                if percent is not None:
                    text = f"{text} ({float(percent):.0f}%)"
                progress_state.update(text=text, changed=True)

        job = api_session.submit_task(settings, callbacks=Callbacks())
        active_job["job"] = job
        try:
            while not job.done:
                if progress_state["changed"]:
                    progress_state["changed"] = False
                    yield f"⏳ {progress_state['text']}"
                time.sleep(0.5)
            result = job.result()
        finally:
            if active_job.get("job") is job:
                active_job["job"] = None
        if result.success and result.generated_files:
            output = str(result.generated_files[0])
            checkpoint = os.path.splitext(output)[0] + ".safetensors"
            latent_note = f" Latent checkpoint: `{checkpoint}`." if save_latent_value and os.path.isfile(checkpoint) else ""
            yield f"✅ Done: `{output}` (also in the Media Generator gallery).{latent_note}"
            return
        if result.cancelled:
            yield "Aborted."
            return
        errors = list(result.errors or [])
        raise gr.Error(str(errors[0] if errors else "WanGP completed without returning an output file."))

    def cancel():
        job = active_job.get("job")
        if job is not None and not job.done:
            job.cancel()

    def model_changed(model_type):
        ref2va = _is_ref2va(model_type)
        return [*_config_dropdown_updates(model_type), gr.update(choices=_attention_choices(model_type), value=""),
                gr.update(visible=ref2va)] + ([gr.update(visible=ref2va)] if refmods_section is not None else [])

    plan_inputs = [source_video, factor, fps_choice, window_size, window_overlap]
    for component in plan_inputs:
        component.change(fn=_plan_text, inputs=plan_inputs, outputs=[plan_info], show_progress="hidden")
    for component in (noise_level, phase_2_steps, phase_2_mode):
        component.change(fn=_phase_2_info, inputs=[phase_2_mode, noise_level, phase_2_steps], outputs=[steps_info], show_progress="hidden", queue=False)
    phase_2_mode.change(fn=lambda mode: [gr.update(visible=mode != PHASE_2_MODE_REGENERATE)] * 2, inputs=[phase_2_mode],
                        outputs=[turbo_controls, multiplier_controls], show_progress="hidden", queue=False)
    model.change(fn=model_changed, inputs=[model], show_progress="hidden",
                 outputs=[*config_dropdowns, override_attention, references] + ([refmods_section] if refmods_section is not None else []))
    for trigger in (model.change, refresh_loras.click, use_loras.change):
        trigger(fn=lambda state_value, model_type: _loras_text(state_value, model_type), inputs=[*state_input, model], outputs=[loras_info], show_progress="hidden")
    refmod_widgets = [component for row in refmod_rows for component in row]
    for component in refmod_widgets:
        component.change(fn=lambda *values: _refmods_status(refmod_module, *values), inputs=refmod_widgets, outputs=[refmods_info],
                         show_progress="hidden", queue=False)
    copy_prompt.click(fn=_read_prompt, inputs=[source_video], outputs=[prompt])
    start.click(fn=run_upscale, inputs=[source_video, factor, model, noise_level, multiplier, prompt, seed,
                                        fps_choice, override_profile, override_attention, tiling, use_loras, window_size, window_overlap,
                                        image_refs_gallery.gallery, phase_2_steps, save_latent, phase_2_mode, *state_input, *config_dropdowns, *reference_videos, *refmod_widgets],
                outputs=[status])
    abort.click(fn=cancel, queue=False)
