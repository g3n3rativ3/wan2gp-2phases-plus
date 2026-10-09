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
FACTOR_CHOICES = [("x1.5", 1.5), ("x2", 2.0), ("x2.5", 2.5), ("x3", 3.0)]
MAX_OUTPUT_SIZE = 4096  # longest output side, in pixels
DEFAULT_FACTOR = 2.0
DEFAULT_NOISE = 0.6
DEFAULT_MULTIPLIER = 0.5
DEFAULT_PROMPT = "."
FPS = 24  # MiniMax H3 default frame rate
FPS_CHOICES = [(f"Model Default ({FPS} fps)", ""), ("Source Video fps", "source")] + [(value, value) for value in ("15", "16", "23", "24", "25", "30", "48", "50")]
FRAMES_MIN, FRAMES_STEP, FRAMES_OFFSET = 107, 17, 5  # MiniMax H3: 107 frames minimum, then 5 + 17n
FRAMES_MAX = 737  # longest video processed (about 30 s at 24 fps), split into sliding windows when longer than the window size
WINDOW_MIN, WINDOW_MAX, WINDOW_STEP, WINDOW_DEFAULT = 124, 481, 17, 481  # H3 FL2VA sliding window sizes
OVERLAP_MIN, OVERLAP_MAX, OVERLAP_STEP, OVERLAP_DEFAULT = 1, 120, 17, 18
SIZE_ALIGNMENT = 32
SOURCE_HEIGHT = 200  # compact video inputs
REFERENCE_HEIGHT = 170
CONFIG_SLOTS = 4  # Text Encoder, Video VAE, DiT Denoising Priority, user configs
WORK_DIR = os.path.join(tempfile.gettempdir(), "wan2gp_h3_upscaler")

INTRO = ("Runs MiniMax H3's **phase 2** on an existing video: the video is encoded at its own resolution, enlarged by the "
         "H3 latent upscaler and refined by phase 2 at the target resolution. The soundtrack of the video is kept. "
         "Videos are processed at 24 fps by default, from 107 frames (4.5 s at 24 fps) up to 737 frames (about 30 s, in sliding windows "
         "beyond the window size), "
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


def _usable_frames(frames_24):
    frames = min(frames_24, FRAMES_MAX)
    if frames < FRAMES_MIN:
        return 0
    return (frames - FRAMES_OFFSET) // FRAMES_STEP * FRAMES_STEP + FRAMES_OFFSET


def _aligned(value):
    return max(SIZE_ALIGNMENT, int(round(value / SIZE_ALIGNMENT)) * SIZE_ALIGNMENT)


def _target_size(width, height, factor):
    """Output size for the factor, with the longest side limited to MAX_OUTPUT_SIZE. Returns (width, height, effective factor)."""
    phase_1_width, phase_1_height = _aligned(width), _aligned(height)
    factor = min(float(factor), MAX_OUTPUT_SIZE / max(phase_1_width, phase_1_height))
    target_width = min(MAX_OUTPUT_SIZE, _aligned(phase_1_width * factor))
    target_height = min(MAX_OUTPUT_SIZE, _aligned(phase_1_height * factor))
    effective = (target_width / phase_1_width + target_height / phase_1_height) / 2.0
    return target_width, target_height, effective


def _plan(path, factor, fps_choice=""):
    metadata = _probe(path)
    width, height = int(metadata.get("display_width") or metadata.get("width")), int(metadata.get("display_height") or metadata.get("height"))
    fps, force_fps = _resolve_fps(fps_choice, metadata)
    frames_available = _frames_at_fps(metadata, fps)
    frames = _usable_frames(frames_available)
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
             f"**Output:** {plan['target_width']}x{plan['target_height']} at {fps} fps"]
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
            reason = f"the maximum is {FRAMES_MAX} frames" if plan["frames_available"] > FRAMES_MAX else "H3 lengths are 5 + 17n frames"
            lines.append(f"The last {dropped} frame(s) ({dropped / fps:.2f} s) are not processed: {reason}.")
    return "<br>".join(lines)


def _prepare_files(path, frames, fps=FPS):
    """Copy the source into a work folder and write its soundtrack as a WAV of exactly the processed duration
    (silence when the video has no audio track)."""
    import numpy as np
    import soundfile as sf
    from shared.utils.audio_video import extract_audio_track_to_wav

    os.makedirs(WORK_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    source_copy = os.path.join(WORK_DIR, f"{stamp}_source{os.path.splitext(path)[1] or '.mp4'}")
    shutil.copyfile(path, source_copy)
    wav_path = os.path.join(WORK_DIR, f"{stamp}_soundtrack.wav")
    duration = frames / fps
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


def _media_generator_refmods(state, model_type):
    """The RefMods selection of the MiniMax H3 RefMods plugin for this model in Media Generator (or None)."""
    wgp = _wgp()
    if wgp is None or not isinstance(state, dict) or not model_type:
        return None
    try:
        settings = wgp.get_model_settings(state, model_type) or {}
    except Exception:
        return None
    value = (settings.get("custom_settings") or {}).get(REFMOD_SETTING)
    return value if value else None


def _refmods_text(state, model_type, use_refmods):
    if not use_refmods or not _is_ref2va(model_type):
        return ""
    value = _media_generator_refmods(state, model_type)
    if value is None:
        return "No RefMod is selected in the **MiniMax H3 RefMods (inline)** panel of Media Generator for this model."
    try:
        import json
        parsed = json.loads(value) if isinstance(value, str) else value
        count = len(parsed.get("rows") or []) if isinstance(parsed, dict) else None
    except Exception:
        count = None
    return "RefMods selection found in Media Generator" + (f" ({count} mod(s))." if count else ".")


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
                   refmods=None):
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
                                    info="Model used for phase 2. Ref2VA can use reference images / videos and RefMods; Turbo finetunes double the Turbo effect.")
        with gr.Column(visible=_is_ref2va(default_model)) as references:
            gr.Markdown("**References (Ref2VA)**: used by phase 2 to keep identities and appearance. Name them in the Phase 2 Prompt "
                        "(`<Picture 1>`, `<Video 1>`...) if the prompt describes them.")
            with gr.Row():
                with gr.Column(scale=2, min_width=260):
                    image_refs_gallery = AdvancedMediaGallery(media_mode="image", height=REFERENCE_HEIGHT, columns=4, label="Reference Images")
                    image_refs_gallery.mount()
                reference_videos = [gr.Video(label=f"Reference Video {index + 1}", sources=["upload"], height=REFERENCE_HEIGHT, min_width=160)
                                    for index in range(REFERENCE_VIDEO_SLOTS)]
            use_refmods = gr.Checkbox(value=False, label="Apply the RefMods selected in Media Generator (MiniMax H3 RefMods inline panel) for this model")
            refmods_info = gr.Markdown("")
        noise_level = gr.Slider(minimum=patch.NOISE_LEVEL_MIN, maximum=1.0, step=0.01, value=DEFAULT_NOISE,
                                label=patch.NOISE_LEVEL_LABEL, show_reset_button=False)
        steps_info = gr.Markdown(patch.phase_2_steps_text(DEFAULT_NOISE))
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
        with gr.Row():
            start = gr.Button("Upscale", variant="primary")
            abort = gr.Button("Abort")

    def run_upscale(video_path, factor_value, model_type, noise_value, multiplier_value, prompt_value, seed_value,
                    fps_value, profile_value, attention_value, tiling_value, use_loras_value, window_value, overlap_value,
                    use_refmods_value, image_refs_value, state_value, *rest):
        # A generator, run directly by WanGP's WebUI session wrapper (it keeps the Gradio session needed to queue the task).
        config_values, reference_video_values = rest[:CONFIG_SLOTS], rest[CONFIG_SLOTS:]
        if not video_path:
            raise gr.Error("Load a video first.")
        if not model_type:
            raise gr.Error("No MiniMax H3 FL2VA or Ref2VA model is available.")
        ref2va = _is_ref2va(model_type)
        factor_value = float(factor_value)
        plan = _plan(video_path, factor_value, fps_value)
        if plan["frames"] == 0:
            raise gr.Error(f"The video is too short: MiniMax H3 needs at least {FRAMES_MIN} frames ({FRAMES_MIN / plan['fps']:.2f} s at {plan['fps']} fps).")
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
            refmods = _media_generator_refmods(state_value, model_type) if use_refmods_value else None
        settings = build_settings(source_copy, wav_path, plan, factor_value, model_type, noise_value, multiplier_value, prompt_value, seed_value,
                                  config=config, override_profile=profile_value, override_attention=attention_value, tiling=bool(tiling_value),
                                  loras=loras, loras_multipliers=loras_multipliers, window_size=window_value, overlap=overlap_value,
                                  ref2va=ref2va, image_refs=image_refs, reference_videos=videos, refmods=refmods)
        print(f"{patch.LOG_PREFIX} H3 Upscaler: {plan['width']}x{plan['height']} -> {settings['resolution']} (x{plan['effective_factor']:.2f}), "
              f"{plan['frames']} frames at {plan['fps']} fps, noise level start {float(noise_value):g}, Turbo multiplier {float(multiplier_value):g}, "
              f"model {model_type}{', tiled phase 2' if tiling_value else ''}{', config ' + config if config else ''}"
              f", sliding window {int(window_value)}/{int(overlap_value)}{', LoRAs ' + str(len(loras)) if loras else ''}"
              + (f", {len(image_refs)} reference image(s), {len(videos)} reference video(s){', RefMods' if refmods else ''}" if ref2va else ""))

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
            yield f"✅ Done: `{result.generated_files[0]}` (also in the Media Generator gallery)."
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
        return [*_config_dropdown_updates(model_type), gr.update(choices=_attention_choices(model_type), value=""),
                gr.update(visible=_is_ref2va(model_type))]

    plan_inputs = [source_video, factor, fps_choice, window_size, window_overlap]
    for component in plan_inputs:
        component.change(fn=_plan_text, inputs=plan_inputs, outputs=[plan_info], show_progress="hidden")
    noise_level.change(fn=patch.phase_2_steps_text, inputs=[noise_level], outputs=[steps_info], show_progress="hidden", queue=False)
    model.change(fn=model_changed, inputs=[model], outputs=[*config_dropdowns, override_attention, references], show_progress="hidden")
    for trigger in (model.change, refresh_loras.click, use_loras.change):
        trigger(fn=lambda state_value, model_type: _loras_text(state_value, model_type), inputs=[*state_input, model], outputs=[loras_info], show_progress="hidden")
    for trigger in (model.change, use_refmods.change):
        trigger(fn=_refmods_text, inputs=[*state_input, model, use_refmods], outputs=[refmods_info], show_progress="hidden")
    copy_prompt.click(fn=_read_prompt, inputs=[source_video], outputs=[prompt])
    start.click(fn=run_upscale, inputs=[source_video, factor, model, noise_level, multiplier, prompt, seed,
                                        fps_choice, override_profile, override_attention, tiling, use_loras, window_size, window_overlap,
                                        use_refmods, image_refs_gallery.gallery, *state_input, *config_dropdowns, *reference_videos],
                outputs=[status])
    abort.click(fn=cancel, queue=False)
