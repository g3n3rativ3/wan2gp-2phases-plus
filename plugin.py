"""2 Phases Plus: extra phase 2 controls for MiniMax H3 two-phase generation.

Adds two controls right below "Phase 2 Noise Level Start" (General tab, Advanced Mode), visible only for MiniMax H3
models when "Two Phases" or "Two Phases with Tiling" is selected:

* Phase 2 Turbo LoRA Multiplier (0 to 1): strength of the Turbo LoRA that WanGP adds automatically to phase 2
  (WanGP forces it to 1.0).
* Phase 2 Prompt: an optional prompt used only by phase 2 instead of the main prompt.

The values are stored in each task's "custom_settings" (keys two_phases_plus_turbo_multiplier and
two_phases_plus_phase2_prompt), so they follow queued tasks, are saved with the settings and written to the metadata.
"""

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from . import h3_phase2_patch as patch
from . import upscaler_tab

UI_STATE_KEY = "two_phases_plus_ui"
TWO_PHASE_VALUES = ("2", "2~")
SAVE_TARGETS = ("state", "edit_state", "settings")
HOOK_MARKER = "_two_phases_plus_hook"

MULTIPLIER_INFO = ("Strength of the Turbo LoRA that WanGP adds automatically to phase 2 (WanGP default: 1.0). "
                   "Lower values let phase 2 stay closer to phase 1; very low values may give a blurry result with only 3 steps.")
PHASE_1_SCALE_CHOICES = [("Output / 2 (WanGP default)", 2.0), ("Output / 1.5", 1.5)]
PHASE_1_SCALE_INFO = ("Phase 1 is rendered at the output resolution divided by this value, then enlarged by the latent upscaler for phase 2. "
                      "/ 1.5 gives phase 1 more detail (larger phase 1, slower and more VRAM in phase 1).")
PROMPT_PLACEHOLDER = ("Empty = phase 2 reuses the main prompt. Write a phase 2 specific prompt here, "
                      "for example a short description without the elements phase 2 should not amplify. "
                      "{prompt} is replaced by the main prompt.")


class TwoPhasesPlusPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = "2 Phases Plus"
        self.version = "2.9.0"
        self.description = "MiniMax H3 Two Phases: phase 2 Turbo LoRA multiplier, phase 2 prompt, extended noise range, and an H3 Upscaler tab."
        self._ui_built = False
        self._hooks_installed = False
        self._wgp = None

    # ------------------------------------------------------------------ setup
    def setup_ui(self):
        patch.extend_noise_level_range()  # before the UI is built, so the slider starts with the extended range
        self.add_tab(tab_id="h3_upscaler", label="H3 Upscaler", component_constructor=self.create_upscaler_tab)
        for component_id in ("state", "guidance_phases", "guidance_phases_row", "refresh_form_trigger", "model_choice_target", "switch_threshold", "num_inference_steps2", "resolution"):
            self.request_component(component_id)

    def post_ui_setup(self, components):
        if self._ui_built:  # post_ui_setup also runs for the hidden Edit tab: the controls live in the main tab only
            return {}
        required = ("state", "guidance_phases", "guidance_phases_row")
        if any(components.get(name) is None for name in required):
            print(f"{patch.LOG_PREFIX} UI components not found, controls not added")
            return {}
        self._ui_built = True
        if not patch.install():
            return {}
        self._install_wgp_hooks()

        state_component = components["state"]
        guidance_phases = components["guidance_phases"]
        refresh_form_trigger = components.get("refresh_form_trigger")
        model_choice_target = components.get("model_choice_target")
        initial_state = state_component.value if isinstance(state_component.value, dict) else {}
        initial_visible = self._controls_visible(initial_state.get("model_type"), guidance_phases.value)
        switch_threshold = components.get("switch_threshold")
        phase_2_steps = components.get("num_inference_steps2")  # WanGP 18+ "Phase 2 Steps" (absent in older versions)
        steps_inputs = [component for component in (switch_threshold, phase_2_steps) if component is not None]
        resolution = components.get("resolution")
        initial_resolution_text = patch.phase_1_resolution_text(resolution.value) if resolution is not None else ""
        initial_steps_text = (patch.phase_2_steps_text(switch_threshold.value, phase_2_steps.value if phase_2_steps is not None else None)
                              if switch_threshold is not None else "")

        def build_controls():
            with gr.Column(visible=initial_visible, elem_id="two_phases_plus_controls") as controls:
                steps_info = gr.Markdown(initial_steps_text, elem_id="two_phases_plus_steps")
                multiplier = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=patch.DEFAULT_MULTIPLIER,
                                       label="Phase 2 Turbo LoRA Multiplier", info=MULTIPLIER_INFO,
                                       show_reset_button=False, elem_id="two_phases_plus_multiplier")
                phase_1_scale = gr.Radio(choices=PHASE_1_SCALE_CHOICES, value=patch.PHASE_1_SCALE_DEFAULT, label="Phase 1 Resolution",
                                         info=PHASE_1_SCALE_INFO, elem_id="two_phases_plus_phase1_scale")
                resolution_info = gr.Markdown(initial_resolution_text, visible=resolution is not None, elem_id="two_phases_plus_resolution")
                phase2_prompt = gr.Textbox(value="", lines=3, max_lines=12, label="Phase 2 Prompt (optional)",
                                           placeholder=PROMPT_PLACEHOLDER, elem_id="two_phases_plus_prompt")

            value_inputs = [state_component, multiplier, phase2_prompt, phase_1_scale]
            for component in (multiplier, phase2_prompt, phase_1_scale):
                component.change(fn=self._store_values, inputs=value_inputs, outputs=None, show_progress="hidden", queue=False)
            guidance_phases.change(fn=self._visibility_from_state, inputs=[state_component, guidance_phases], outputs=[controls],
                                   show_progress="hidden")
            if model_choice_target is not None:
                model_choice_target.change(fn=self._visibility_from_target, inputs=[model_choice_target, guidance_phases],
                                           outputs=[controls], show_progress="hidden")
            if switch_threshold is not None:
                for component in steps_inputs:
                    component.change(fn=patch.phase_2_steps_text, inputs=steps_inputs, outputs=[steps_info],
                                     show_progress="hidden", queue=False)
            if resolution is not None:  # any resolution change (presets, megapixels, model switch, loaded settings), no render needed
                for component in (resolution, phase_1_scale):
                    component.change(fn=patch.phase_1_resolution_text, inputs=[resolution, phase_1_scale], outputs=[resolution_info],
                                     show_progress="hidden", queue=False)
            if refresh_form_trigger is not None:  # settings loaded from a video, an image or a settings file
                refresh_form_trigger.change(fn=self._load_from_settings, inputs=[state_component],
                                            outputs=[multiplier, phase2_prompt, phase_1_scale, controls], show_progress="hidden")
            return controls

        self.insert_after("guidance_phases_row", build_controls)
        return {}

    # ------------------------------------------------------------- wgp hooks
    def _install_wgp_hooks(self):
        """Keep the two values inside custom_settings: WanGP otherwise drops undeclared custom settings."""
        set_global_func = getattr(self, "_set_wgp_global_func", None)
        wgp_globals = getattr(set_global_func, "__globals__", None)
        if not isinstance(wgp_globals, dict) or "prepare_inputs_dict" not in wgp_globals or "collect_custom_settings_from_inputs" not in wgp_globals:
            print(f"{patch.LOG_PREFIX} WanGP globals not found, the values will be taken from the UI controls at generation time")
            return
        self._wgp = wgp_globals
        self._ensure_wgp_hooks(verbose=True)

    @staticmethod
    def _is_hooked(function):
        seen = 0
        while function is not None and seen < 32:  # follow the __wrapped__ chain left by other plugins too
            if getattr(function, HOOK_MARKER, False):
                return True
            function = getattr(function, "__wrapped__", None)
            seen += 1
        return False

    def _ensure_wgp_hooks(self, verbose=False):
        """(Re)install the hooks. Another plugin may replace these WanGP functions after this plugin hooked them;
        the hooks are then installed again on top of the current functions."""
        wgp_globals = self._wgp
        if not isinstance(wgp_globals, dict):
            return
        reinstalled = []
        current_collect = wgp_globals.get("collect_custom_settings_from_inputs")
        if callable(current_collect) and not self._is_hooked(current_collect):
            wgp_globals["collect_custom_settings_from_inputs"] = self._make_collect_hook(current_collect)
            reinstalled.append("collect_custom_settings_from_inputs")
        current_prepare = wgp_globals.get("prepare_inputs_dict")
        if callable(current_prepare) and not self._is_hooked(current_prepare):
            wgp_globals["prepare_inputs_dict"] = self._make_prepare_hook(current_prepare)
            reinstalled.append("prepare_inputs_dict")
        if reinstalled and (verbose or self._hooks_installed):
            print(f"{patch.LOG_PREFIX} {'hooked' if not self._hooks_installed else 're-hooked'} {', '.join(reinstalled)}")
        self._hooks_installed = True

    def _make_collect_hook(self, original_collect):
        def collect_custom_settings_from_inputs(model_def, inputs, strict=False):
            parsed, error = original_collect(model_def, inputs, strict=strict)
            if error is not None:
                return parsed, error
            existing = inputs.get("custom_settings") if isinstance(inputs, dict) else None
            if isinstance(existing, dict):
                kept = {key: existing[key] for key in patch.PRESERVED_KEYS if key in existing}
                if kept:
                    parsed = {**(parsed or {}), **kept}
            return parsed, error

        collect_custom_settings_from_inputs.__wrapped__ = original_collect
        setattr(collect_custom_settings_from_inputs, HOOK_MARKER, True)
        return collect_custom_settings_from_inputs

    def _make_prepare_hook(self, original_prepare):
        def prepare_inputs_dict(target, inputs, model_type=None, model_filename=None):
            upscale_refmods = None
            try:
                existing = inputs.get("custom_settings") if isinstance(inputs, dict) else None
                if isinstance(existing, dict) and existing.get(patch.UPSCALE_SOURCE_KEY):
                    # H3 Upscaler task: its RefMods come from the tab's own panel, not from the Media Generator selection
                    upscale_refmods = (existing.get(upscaler_tab.REFMOD_SETTING) or "",)
            except Exception:
                upscale_refmods = None
            try:
                if target in SAVE_TARGETS and isinstance(inputs, dict):
                    state = inputs.get("state")
                    current_model = model_type or (self._call("get_state_model_type", state) if isinstance(state, dict) else None)
                    if isinstance(state, dict) and self._is_h3_two_phase_model(current_model):
                        values = self._values_from_state(state)
                        existing = inputs.get("custom_settings")
                        inputs["custom_settings"] = {**(existing if isinstance(existing, dict) else {}), **values}
                    patch.install(verbose=False)  # queue time: make sure the pipeline patch is still in place
            except Exception as error:
                print(f"{patch.LOG_PREFIX} could not attach the phase 2 values: {error}")
            result = original_prepare(target, inputs, model_type, model_filename)
            if upscale_refmods is not None and isinstance(result, dict):
                try:
                    settings = dict(result.get("custom_settings") or {})
                    if upscale_refmods[0]:
                        settings[upscaler_tab.REFMOD_SETTING] = upscale_refmods[0]
                    else:
                        settings.pop(upscaler_tab.REFMOD_SETTING, None)
                    result["custom_settings"] = settings
                except Exception:
                    pass
            return result

        prepare_inputs_dict.__wrapped__ = original_prepare
        setattr(prepare_inputs_dict, HOOK_MARKER, True)
        return prepare_inputs_dict

    def create_upscaler_tab(self, api_session):
        upscaler_tab.create_tab(self, api_session)

    def ensure_hooks(self):
        """Make sure the settings hooks are in place (used before queuing an H3 Upscaler task)."""
        if self._wgp is None:
            self._install_wgp_hooks()
        else:
            self._ensure_wgp_hooks()

    def _call(self, name, *args):
        function = (self._wgp or {}).get(name)
        return function(*args) if callable(function) else None

    # ------------------------------------------------------------- helpers
    def _is_h3_two_phase_model(self, model_type):
        if not model_type:
            return False
        try:
            model_def = self._call("get_model_def", model_type) or {}
            base_model_type = self._call("get_base_model_type", model_type) or ""
        except Exception:
            return False
        return str(base_model_type).startswith("minimax_h3") and bool(model_def.get("phase_2_spatial_tiling", False))

    def _controls_visible(self, model_type, guidance_phases_value):
        return str(guidance_phases_value) in TWO_PHASE_VALUES and self._is_h3_two_phase_model(model_type)

    @staticmethod
    def _values_from_state(state):
        values = state.get(UI_STATE_KEY) if isinstance(state, dict) else None
        values = values if isinstance(values, dict) else {}
        return {patch.MULTIPLIER_KEY: patch.parse_multiplier(values.get(patch.MULTIPLIER_KEY, patch.DEFAULT_MULTIPLIER)),
                patch.PROMPT_KEY: patch.parse_prompt(values.get(patch.PROMPT_KEY, "")),
                patch.PHASE_1_SCALE_KEY: patch.parse_phase_1_scale(values.get(patch.PHASE_1_SCALE_KEY, patch.PHASE_1_SCALE_DEFAULT))}

    # ------------------------------------------------------------- events
    def _store_values(self, state, multiplier, phase2_prompt, phase_1_scale=None):
        values = {patch.MULTIPLIER_KEY: patch.parse_multiplier(multiplier),
                  patch.PROMPT_KEY: phase2_prompt if isinstance(phase2_prompt, str) else "",
                  patch.PHASE_1_SCALE_KEY: patch.parse_phase_1_scale(phase_1_scale)}
        patch.UI_VALUES.update(values)
        if isinstance(state, dict):
            state[UI_STATE_KEY] = values
        self._ensure_wgp_hooks()
        patch.install(verbose=False)

    def _visibility_from_state(self, state, guidance_phases_value):
        self._ensure_wgp_hooks()
        patch.install(verbose=False)
        model_type = self._call("get_state_model_type", state) if isinstance(state, dict) else None
        return gr.update(visible=self._controls_visible(model_type, guidance_phases_value))

    def _visibility_from_target(self, model_target, guidance_phases_value):
        model_type = self._call("_model_choice_target_model_type", model_target) if model_target else None
        return gr.update(visible=self._controls_visible(model_type, guidance_phases_value))

    def _load_from_settings(self, state):
        if not isinstance(state, dict):
            return gr.update(), gr.update(), gr.update(), gr.update()
        model_type = self._call("get_state_model_type", state)
        settings = self._call("get_model_settings", state, model_type) or {}
        custom_settings = settings.get("custom_settings") if isinstance(settings, dict) else None
        visible = gr.update(visible=self._controls_visible(model_type, settings.get("guidance_phases") if isinstance(settings, dict) else None))
        if not isinstance(custom_settings, dict) or not any(key in custom_settings for key in patch.SETTING_KEYS):
            return gr.update(), gr.update(), gr.update(), visible
        multiplier = patch.parse_multiplier(custom_settings.get(patch.MULTIPLIER_KEY, patch.DEFAULT_MULTIPLIER))
        phase2_prompt = patch.parse_prompt(custom_settings.get(patch.PROMPT_KEY, ""))
        phase_1_scale = patch.parse_phase_1_scale(custom_settings.get(patch.PHASE_1_SCALE_KEY, patch.PHASE_1_SCALE_DEFAULT))
        state[UI_STATE_KEY] = {patch.MULTIPLIER_KEY: multiplier, patch.PROMPT_KEY: phase2_prompt, patch.PHASE_1_SCALE_KEY: phase_1_scale}
        patch.UI_VALUES.update(state[UI_STATE_KEY])
        return gr.update(value=multiplier), gr.update(value=phase2_prompt), gr.update(value=phase_1_scale), visible
