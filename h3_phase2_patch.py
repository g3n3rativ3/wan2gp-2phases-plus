"""Runtime patch of the MiniMax H3 pipeline used by the 2 Phases Plus plugin.

Two behaviours are added to H3 two-phase generation ("Two Phases" and "Two Phases with Tiling"):

1. Phase 2 Turbo LoRA multiplier: WanGP automatically adds the LightX2V FL2V 4-step v0.1 Turbo LoRA for phase 2 and
   forces it to 1.0. When the user picks another value, the multiplier of that LoRA is replaced for phase 2 only.
2. Phase 2 prompt: phase 2 re-encodes the prompt before refining the upscaled latent. When the user provides a phase 2
   prompt, it replaces the main prompt for that encoding only (phase 1 and the optional audio refinement keep the main
   prompt). "{prompt}" inside the phase 2 prompt is replaced by the main prompt.

Both values travel with each task inside its "custom_settings" dictionary, so queued tasks keep their own values and the
values are written to the output metadata.

Detection of phase 2 relies on the pipeline's own sequence: after the latent upscaler, the pipeline activates the phase 2
LoRA multipliers (update_loras_slists called with phase_switch_step=0, which only happens for phase 2) and then encodes
the phase 2 prompt. No pipeline source file is modified.
"""

import os
import threading

MULTIPLIER_KEY = "two_phases_plus_turbo_multiplier"
PROMPT_KEY = "two_phases_plus_phase2_prompt"
SETTING_KEYS = (MULTIPLIER_KEY, PROMPT_KEY)
# H3 Upscaler tab: the source video replaces phase 1, and the phase 1 -> phase 2 scale is the chosen factor
UPSCALE_SOURCE_KEY = "two_phases_plus_upscale_source"
UPSCALE_FACTOR_KEY = "two_phases_plus_upscale_factor"
PRESERVED_KEYS = SETTING_KEYS + (UPSCALE_SOURCE_KEY, UPSCALE_FACTOR_KEY)
DEFAULT_MULTIPLIER = 0.5  # plugin default of the phase 2 Turbo LoRA multiplier (strength recommended by the LoRA's profile)
WANGP_MULTIPLIER = 1.0  # what WanGP itself uses: no change needed at that value
NOISE_LEVEL_DEFAULT = 0.6  # plugin default of "Phase 2 Noise Level Start" (2 phase 2 steps)
WANGP_NOISE_LEVEL_DEFAULT = 0.9035  # WanGP's default, replaced by NOISE_LEVEL_DEFAULT in default settings only
LOG_PREFIX = "[2 Phases Plus]"
PATCH_MARKER = "_two_phases_plus_patch"
# Lowest "Phase 2 Noise Level Start" offered by the plugin (WanGP's own minimum is 0.7).
NOISE_LEVEL_MIN = 0.01
# Latent upscaler temporal chunk size forced by the plugin. WanGP splits the latent into chunks of 16 latent frames with
# only 2 frames of context and no blending, which creates one-frame flashes at the chunk boundaries once phase 2 starts at
# a low noise level. 1024 latent frames (about 70 s at 24 fps) means a single pass for any H3 video (30 s max).
UPSCALER_TEMPORAL_CHUNK = 1024
# Fixed intermediate sigmas of WanGP's phase 2 schedule (start, 0.6316, 0.3158, 0)
PHASE_2_FIXED_SIGMAS = (0.6316, 0.3158)
NOISE_LEVEL_LABEL = "Phase 2 Noise Level Start (≤ 0.3158: 1 step | ≤ 0.6316: 2 steps | > 0.6316: 3 steps)"

_state = threading.local()
_installed = False
# Last values set in the UI controls. Used when a task does not carry the values in its custom_settings
# (for instance when another plugin replaced WanGP's settings functions after this plugin hooked them).
UI_VALUES = {MULTIPLIER_KEY: DEFAULT_MULTIPLIER, PROMPT_KEY: ""}


def _current():
    return getattr(_state, "run", None)


def parse_multiplier(value):
    try:
        multiplier = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MULTIPLIER
    return min(1.0, max(0.0, multiplier))


def parse_prompt(value):
    return value.strip() if isinstance(value, str) else ""


def _lora_name(lora):
    return os.path.basename(str(lora).split("|", 1)[0])


def _is_patched(function):
    seen = 0
    while function is not None and seen < 32:  # follow the __wrapped__ chain left by other plugins too
        if getattr(function, PATCH_MARKER, False):
            return True
        function = getattr(function, "__wrapped__", None)
        seen += 1
    return False


def _mark(wrapper, original):
    wrapper.__wrapped__ = original
    wrapper.__name__ = getattr(original, "__name__", wrapper.__name__)
    wrapper.__doc__ = getattr(original, "__doc__", None)
    setattr(wrapper, PATCH_MARKER, True)
    return wrapper


def _make_generate(original_generate):
    def generate(self, *args, **kwargs):
        custom_settings = kwargs.get("custom_settings")
        custom_settings = custom_settings if isinstance(custom_settings, dict) else {}
        from_task = any(key in custom_settings for key in SETTING_KEYS)
        source = custom_settings if from_task else UI_VALUES
        multiplier = parse_multiplier(source.get(MULTIPLIER_KEY, DEFAULT_MULTIPLIER))
        phase2_prompt = parse_prompt(source.get(PROMPT_KEY, ""))
        try:
            two_phase = int(str(kwargs.get("guide_phases", 1)).strip("~") or 1) > 1
        except ValueError:
            two_phase = False
        if two_phase:
            print(f"{LOG_PREFIX} values {'from the task settings' if from_task else 'from the UI controls'}: "
                  f"phase 2 Turbo LoRA multiplier {multiplier:g}, phase 2 prompt {'set (' + str(len(phase2_prompt)) + ' characters)' if phase2_prompt else 'empty'}")
        schedule_override = None
        if two_phase:
            schedule_override = _phase_2_schedule_for(kwargs.get("switch_threshold"))
        previous = _current()
        _state.run = {
            "multiplier": multiplier,
            "prompt": phase2_prompt,
            "loras_selected": list(kwargs.get("loras_selected") or ()),
            "phase_2": False,
            "prompt_done": False,
        }
        if two_phase:
            _force_single_pass(getattr(self, "latent_upscaler", None), verbose=True)
        h3_pipeline = _pipeline_module()
        original_schedule = getattr(h3_pipeline, "H3_PHASE_2_SIGMAS", None) if h3_pipeline is not None else None
        upscale_source = custom_settings.get(UPSCALE_SOURCE_KEY) if two_phase else None
        original_scale = getattr(h3_pipeline, "H3_TWO_PHASE_SCALE", None) if h3_pipeline is not None else None
        restore_forward = None
        try:
            if upscale_source:
                factor = float(custom_settings.get(UPSCALE_FACTOR_KEY, 2.0) or 2.0)
                h3_pipeline.H3_TWO_PHASE_SCALE = factor
                try:
                    source_fps = float(kwargs.get("fps", 24) or 24)
                except (TypeError, ValueError):
                    source_fps = 24.0
                start_frame, frame_count = _source_window(kwargs)
                restore_forward = _install_source_phase_1(self, str(upscale_source), frame_count, source_fps, start_frame)
                _state.run["upscale"] = True
                print(f"{LOG_PREFIX} H3 Upscaler: phase 1 replaced by the source video '{os.path.basename(str(upscale_source))}', upscale x{factor:g}")
            if schedule_override is not None and original_schedule is not None:
                h3_pipeline.H3_PHASE_2_SIGMAS = schedule_override
                print(f"{LOG_PREFIX} phase 2 noise level start {schedule_override[0]:g} is at or below {original_schedule[1]:g}: "
                      f"phase 2 sigmas {', '.join(f'{value:g}' for value in schedule_override)} ({len(schedule_override) - 1} step(s))")
            return original_generate(self, *args, **kwargs)
        finally:
            if restore_forward is not None:
                restore_forward()
            if upscale_source and original_scale is not None:
                h3_pipeline.H3_TWO_PHASE_SCALE = original_scale
            if schedule_override is not None and original_schedule is not None:
                h3_pipeline.H3_PHASE_2_SIGMAS = original_schedule
            _state.run = previous

    return _mark(generate, original_generate)


def _make_update_loras_slists(original_update_loras_slists, is_required_turbo):
    def update_loras_slists(trans, slists_dict, num_inference_steps, phase_switch_step=None, phase_switch_step2=None):
        run = _current()
        if run is not None and phase_switch_step == 0 and isinstance(slists_dict, dict):
            run["phase_2"] = True
            if run.get("upscale") and not run.get("upscale_used"):
                print(f"{LOG_PREFIX} WARNING H3 Upscaler: the source video did not replace phase 1 (the transformer call was not intercepted); the result will not follow the source")
            multiplier = run["multiplier"]
            if multiplier != WANGP_MULTIPLIER:
                turbo_index = None
                if is_required_turbo is not None:
                    turbo_index = next((index for index, lora in enumerate(run["loras_selected"]) if is_required_turbo(_lora_name(lora))), None)
                if turbo_index is None or turbo_index >= len(slists_dict.get("phase2", ())):
                    print(f"{LOG_PREFIX} phase 2: no built-in Turbo LoRA found, multiplier {multiplier:g} not applied")
                else:
                    slists_dict = dict(slists_dict)
                    slists_dict["phase2"] = list(slists_dict["phase2"])
                    slists_dict["shared"] = list(slists_dict["shared"])
                    slists_dict["phase2"][turbo_index] = float(multiplier)
                    slists_dict["shared"][turbo_index] = False
                    print(f"{LOG_PREFIX} phase 2: Turbo LoRA '{_lora_name(run['loras_selected'][turbo_index])}' multiplier set to {multiplier:g} (default 1.0)")
        return original_update_loras_slists(trans, slists_dict, num_inference_steps, phase_switch_step=phase_switch_step, phase_switch_step2=phase_switch_step2)

    return _mark(update_loras_slists, original_update_loras_slists)


def _make_encode_prompt(original_encode_prompt):
    def _encode_prompt(self, prompt, presentation):
        run = _current()
        if run is not None and run["phase_2"] and not run["prompt_done"]:
            run["prompt_done"] = True  # only the phase 2 encoding; a later audio refinement keeps the main prompt
            if run["prompt"]:
                main_prompt = prompt if isinstance(prompt, str) else ""
                prompt = run["prompt"].replace("{prompt}", main_prompt)
                print(f"{LOG_PREFIX} phase 2: using the Phase 2 Prompt ({len(prompt)} characters)")
        return original_encode_prompt(self, prompt, presentation)

    return _mark(_encode_prompt, original_encode_prompt)


_DEFAULT_PHASE_2_SIGMAS = None


def _force_single_pass(upscaler, verbose=False):
    if upscaler is None or not hasattr(upscaler, "temporal_chunk_size"):
        return
    try:
        current = int(upscaler.temporal_chunk_size)
    except (TypeError, ValueError):
        current = 0
    if current < UPSCALER_TEMPORAL_CHUNK:
        upscaler.temporal_chunk_size = UPSCALER_TEMPORAL_CHUNK
        if verbose:
            print(f"{LOG_PREFIX} latent upscaler: temporal chunking disabled (single pass up to {UPSCALER_TEMPORAL_CHUNK} latent frames, WanGP default {current})")


def _make_upscaler_forward(original_forward):
    def forward(self, *args, **kwargs):
        # H3 Upscaler tab only, x1 (refiner): same size in and out, so the latent is returned unchanged, as the original
        # ComfyUI node does, instead of running the network at a scale it was not meant for. Media Generator is not affected.
        run = _current()
        latent = args[0] if args else kwargs.get("latent")
        target_size = kwargs.get("target_size", args[2] if len(args) > 2 else None)
        if (run is not None and run.get("upscale") and latent is not None and target_size is not None
                and tuple(int(size) for size in target_size) == tuple(latent.shape[2:])):
            print(f"{LOG_PREFIX} latent upscaler skipped: same size in and out (x1 refiner)")
            return latent
        _force_single_pass(self)
        return original_forward(self, *args, **kwargs)

    return _mark(forward, original_forward)


def _pipeline_module():
    try:
        from models.minimax_h3 import pipeline as h3_pipeline
    except Exception:
        return None
    return h3_pipeline


def _phase_2_schedule_for(switch_threshold):
    """WanGP builds the phase 2 schedule as (noise level start, 0.6316, 0.3158, 0). Below 0.6316 that schedule would
    go back up in noise, so only the fixed sigmas lower than the start are kept. Returns None when no change is needed."""
    global _DEFAULT_PHASE_2_SIGMAS
    try:
        start = float(switch_threshold)
    except (TypeError, ValueError):
        return None
    h3_pipeline = _pipeline_module()
    if h3_pipeline is None:
        return None
    if _DEFAULT_PHASE_2_SIGMAS is None:
        _DEFAULT_PHASE_2_SIGMAS = tuple(float(value) for value in h3_pipeline.H3_PHASE_2_SIGMAS)
    intermediate = _DEFAULT_PHASE_2_SIGMAS[1:-1]
    if not intermediate or start > intermediate[0] or start <= 0.0:
        return None
    return (start, *[value for value in intermediate if value < start], 0.0)


def _load_source_frames(path, frame_count, fps=24, start_frame=0):
    """The source video as a CTHW float tensor in [-1, 1], resampled to the generation fps, exactly frame_count frames
    (the last frame is repeated if the source is shorter)."""
    import torch
    from shared.utils.utils import guide_to_float
    from shared.utils.video_decode import decode_video_frames_ffmpeg
    frames = decode_video_frames_ffmpeg(path, int(start_frame), frame_count, target_fps=fps, bridge="torch")
    if not torch.is_tensor(frames):
        frames = torch.as_tensor(frames)
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise RuntimeError(f"H3 Upscaler: could not read frames from {path}")
    if frames.shape[-1] in (3, 4):  # T H W C -> C T H W
        frames = frames[..., :3].permute(3, 0, 1, 2)
    frames = frames[:, :frame_count]
    if frames.shape[1] < frame_count:
        frames = torch.cat((frames, frames[:, -1:].expand(-1, frame_count - frames.shape[1], -1, -1)), dim=1)
    return guide_to_float(frames.contiguous())


def _source_window(generate_kwargs):
    """(first source frame, frame count) of the latent that phase 1 produces for this sliding window, replicating how the
    H3 pipeline splits a window into history frames (conditions) and target frames."""
    from shared.utils.frame_scheduler import floor_frame_count, normalize_frame_count, normalize_overlap
    frame_num = int(generate_kwargs.get("frame_num", 0) or 0)
    window_start = int(generate_kwargs.get("window_start_frame_no", 0) or 0)
    prefix, _ = normalize_overlap(int(generate_kwargs.get("prefix_frames_count", 0) or 0), 17, 1)
    input_video = generate_kwargs.get("input_video")
    continuation_frames = int(input_video.shape[1]) if input_video is not None and getattr(input_video, "ndim", 0) == 4 and prefix > 0 else (
        int(input_video.shape[0]) if input_video is not None and getattr(input_video, "ndim", 0) == 3 and prefix > 0 else 0)
    continuation_count = min(prefix, continuation_frames) if continuation_frames and generate_kwargs.get("image_start") is None else 0
    if continuation_count and continuation_count < prefix:
        continuation_count = floor_frame_count(continuation_count, 1, 17, 1)
    history = continuation_count - 1 if continuation_count > 1 else 0
    return window_start + history, normalize_frame_count(max(1, frame_num - history), 5, 17, 5)


def _install_source_phase_1(pipeline, source_path, frame_num, fps=24, start_frame=0):
    """Replace the transformer during phase 1 by an exact predictor of the encoded source video: with one Euler step from
    sigma 1 to 0, the phase 1 result is then the source latent itself (no model computation). Phase 2 uses the real model."""
    import torch
    from models.minimax_h3 import pipeline as h3_pipeline
    from shared.utils.frame_scheduler import normalize_frame_count

    transformer = pipeline.transformer
    had_instance_forward = "forward" in transformer.__dict__
    previous_forward = transformer.__dict__.get("forward")
    real_forward = transformer.forward
    cache = {}

    def encode_source(video_shape, device):
        _, _, latent_t, latent_h, latent_w = video_shape
        frame_count = normalize_frame_count(max(1, frame_num), 5, 17, 5)
        frames = _load_source_frames(source_path, frame_count, fps, start_frame)
        frames = h3_pipeline._resize_video(frames, latent_h * 16, latent_w * 16)
        pipeline._use_shared_components()
        try:
            latents = h3_pipeline._encode_video_source(pipeline.vae, frames, pipeline.device).float()
        finally:
            pipeline._use_transformer()
        if latents.shape[2] < latent_t:
            latents = torch.cat((latents, latents[:, :, -1:].expand(-1, -1, latent_t - latents.shape[2], -1, -1)), dim=2)
        latents = latents[:, :, :latent_t]
        if tuple(latents.shape[-2:]) != (latent_h, latent_w):
            raise RuntimeError(f"H3 Upscaler: encoded source latent {tuple(latents.shape)} does not match phase 1 latent {tuple(video_shape)}")
        print(f"{LOG_PREFIX} H3 Upscaler: source encoded at {latent_w * 16}x{latent_h * 16}, frames {start_frame}-{start_frame + frame_count - 1} at {fps:g} fps, latent {tuple(latents.shape)}")
        return latents.to(device)

    def forward(*args, **kwargs):
        run = _current()
        if run is None or run.get("phase_2"):
            return real_forward(*args, **kwargs)
        video_x, audio_x, sigma_video = args[0], args[1], args[2]
        if "x0" not in cache:
            cache["x0"] = encode_source(tuple(video_x.shape), video_x.device)
        if run is not None:
            run["upscale_used"] = True
        sigma = float(sigma_video.flatten()[0])
        sigma = sigma if sigma > 0 else 1.0
        velocity = (cache["x0"].to(device=video_x.device, dtype=torch.float32) - video_x.float()).div_(sigma)
        return velocity.to(video_x.dtype), torch.zeros_like(audio_x)

    transformer.forward = forward

    def restore():
        if had_instance_forward:
            transformer.forward = previous_forward
        else:
            transformer.__dict__.pop("forward", None)
        cache.clear()

    return restore


def _find_wgp_module():
    import sys
    main = sys.modules.get("__main__")
    if main is not None and isinstance(getattr(main, "models_def", None), dict) and callable(getattr(main, "init_model_def", None)):
        return main
    for module in list(sys.modules.values()):
        if module is not None and isinstance(getattr(module, "models_def", None), dict) and callable(getattr(module, "init_model_def", None)):
            return module
    return None


def phase_2_schedule(start):
    """Phase 2 sigmas used for a given noise level start (WanGP's schedule above 0.6316, the plugin's below)."""
    try:
        start = float(start)
    except (TypeError, ValueError):
        return None
    intermediate = (_DEFAULT_PHASE_2_SIGMAS or (0.9035, *PHASE_2_FIXED_SIGMAS, 0.0))[1:-1]
    if intermediate and start > intermediate[0]:
        return (start, *intermediate, 0.0)
    return (start, *[value for value in intermediate if value < start], 0.0)


def phase_2_steps_text(start):
    schedule = phase_2_schedule(start)
    if schedule is None:
        return ""
    steps = len(schedule) - 1
    return f"**Phase 2: {steps} step{'s' if steps > 1 else ''}** (sigmas {', '.join(f'{value:g}' for value in schedule)})"


def _lower_noise_min(model_def):
    setting = model_def.get("switch_threshold") if isinstance(model_def, dict) else None
    if not isinstance(setting, dict):
        return False
    changed = False
    if float(setting.get("min", NOISE_LEVEL_MIN)) > NOISE_LEVEL_MIN:
        setting = {**setting, "min": NOISE_LEVEL_MIN}
        changed = True
    if setting.get("label") != NOISE_LEVEL_LABEL:
        setting = {**setting, "label": NOISE_LEVEL_LABEL}
        changed = True
    if changed:
        model_def["switch_threshold"] = setting
    return changed


def _apply_plugin_noise_default(model_def, ui_defaults):
    """Default settings of a two-phase MiniMax H3 model: WanGP's untouched 0.9035 becomes the plugin's 0.6.
    A value set by the user (anything other than WanGP's default) is kept."""
    if not isinstance(ui_defaults, dict) or not isinstance(model_def, dict) or not model_def.get("phase_2_spatial_tiling", False):
        return
    if not str(model_def.get("architecture", "")).startswith("minimax_h3"):
        return
    value = ui_defaults.get("switch_threshold")
    try:
        if value is None or abs(float(value) - WANGP_NOISE_LEVEL_DEFAULT) < 1e-6:
            ui_defaults["switch_threshold"] = NOISE_LEVEL_DEFAULT
    except (TypeError, ValueError):
        pass


def extend_noise_level_range():
    """Lower the minimum of "Phase 2 Noise Level Start" for the MiniMax H3 models: in the model definitions already
    loaded, and in the H3 handler so that rebuilt definitions keep it."""
    try:
        from models.minimax_h3.minimax_h3_handler import family_handler
    except Exception as error:
        print(f"{LOG_PREFIX} MiniMax H3 handler not available, noise level range unchanged: {error}")
        return False
    original_query = family_handler.query_model_def
    if not _is_patched(original_query):
        def query_model_def(base_model_type, model_def):
            result = original_query(base_model_type, model_def)
            _lower_noise_min(result)
            return result
        family_handler.query_model_def = staticmethod(_mark(query_model_def, original_query))
    original_defaults = family_handler.update_default_settings
    if not _is_patched(original_defaults):
        def update_default_settings(base_model_type, model_def, ui_defaults):
            result = original_defaults(base_model_type, model_def, ui_defaults)
            _apply_plugin_noise_default(model_def, ui_defaults)
            return result
        family_handler.update_default_settings = staticmethod(_mark(update_default_settings, original_defaults))
    wgp = _find_wgp_module()
    if wgp is not None and callable(getattr(wgp, "get_default_settings", None)) and not _is_patched(wgp.get_default_settings):
        original_get_default_settings = wgp.get_default_settings

        def get_default_settings(model_type):
            ui_defaults = original_get_default_settings(model_type)
            try:
                _apply_plugin_noise_default(wgp.get_model_def(model_type), ui_defaults)
            except Exception:
                pass
            return ui_defaults
        wgp.get_default_settings = _mark(get_default_settings, original_get_default_settings)
    count = 0
    if wgp is not None:
        for model_def in wgp.models_def.values():
            if isinstance(model_def, dict) and str(model_def.get("architecture", "")).startswith("minimax_h3") and _lower_noise_min(model_def):
                count += 1
    print(f"{LOG_PREFIX} Phase 2 Noise Level Start minimum lowered to {NOISE_LEVEL_MIN:g} ({count} loaded MiniMax H3 model definition(s) updated); "
          f"defaults: noise level start {NOISE_LEVEL_DEFAULT:g}, phase 2 Turbo LoRA multiplier {DEFAULT_MULTIPLIER:g}")
    return True


def install(verbose=True):
    """Patch the H3 pipeline. Safe to call any number of times: only the missing patches are (re)installed, on top of
    whatever another plugin may have installed in the meantime."""
    global _installed
    try:
        from models.minimax_h3 import pipeline as h3_pipeline
    except Exception as error:  # the H3 family may be missing in some installations
        if verbose:
            print(f"{LOG_PREFIX} MiniMax H3 pipeline not available, plugin inactive: {error}")
        return False

    pipeline_class = h3_pipeline.MiniMaxH3Pipeline
    patched = []
    if not _is_patched(pipeline_class.generate):
        pipeline_class.generate = _make_generate(pipeline_class.generate)
        patched.append("generate")
    if not _is_patched(pipeline_class._encode_prompt):
        pipeline_class._encode_prompt = _make_encode_prompt(pipeline_class._encode_prompt)
        patched.append("_encode_prompt")
    if not _is_patched(h3_pipeline.update_loras_slists):
        h3_pipeline.update_loras_slists = _make_update_loras_slists(h3_pipeline.update_loras_slists, getattr(h3_pipeline, "_is_required_h3_turbo", None))
        patched.append("update_loras_slists")
    try:
        from models.minimax_h3.latent_upscaler import MiniMaxH3LatentUpscaler
    except Exception:
        MiniMaxH3LatentUpscaler = None
    if MiniMaxH3LatentUpscaler is not None and not _is_patched(MiniMaxH3LatentUpscaler.forward):
        MiniMaxH3LatentUpscaler.forward = _make_upscaler_forward(MiniMaxH3LatentUpscaler.forward)
        patched.append("latent upscaler single pass")
    if patched:
        print(f"{LOG_PREFIX} MiniMax H3 pipeline {'re-' if _installed else ''}patched ({', '.join(patched)})")
    _installed = True
    return True
