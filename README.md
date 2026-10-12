# 2 Phases Plus (WanGP plugin)

Extra phase 2 controls for **MiniMax H3** two-phase generation in WanGP, and an **H3 Upscaler** tab that runs H3's phase 2 on an existing video.

When **Two Phases** or **Two Phases with Tiling** is selected, two controls appear right below **Phase 2 Noise Level Start**:

| Control | Range | Default | Effect |
|---|---|---|---|
| **Phase 2 Turbo LoRA Multiplier** | 0 to 1 | 0.5 | Strength of the Turbo LoRA that WanGP adds automatically to phase 2. WanGP itself always forces it to 1.0; 0.5 is the strength recommended by that LoRA's profile. |
| **Phase 2 Prompt (optional)** | text | empty | Prompt used only by phase 2. Empty means phase 2 reuses the main prompt, which is WanGP's normal behaviour. |

The plugin also extends **Phase 2 Noise Level Start** itself:
- Its default becomes **0.6** instead of WanGP's 0.9035, and on WanGP 18+ the default of **Phase 2 Steps** becomes **2** instead of 3 (sigmas 0.6, 0.3, 0), in the default settings of the two-phase MiniMax H3 models. This applies to factory settings and to saved default settings that still hold WanGP's untouched values (0.9035 / 3 steps); a value you set yourself is kept, and settings loaded from a video or a settings file are not changed.
- Its minimum goes down from 0.7 (WanGP) to **0.01**.
- Its label shows the step limits: `Phase 2 Noise Level Start (≤ 0.3158: 1 step | ≤ 0.6316: 2 steps | > 0.6316: 3 steps)`.
- A line above the two controls shows, live, the number of phase 2 steps and the exact sigmas for the current value, for example **Phase 2: 2 steps** (sigmas 0.5, 0.3158, 0).

It also fixes a flash artifact of the latent upscaler (always on, no setting):
- WanGP's latent upscaler processes the latent in chunks of 16 latent frames, with only 2 frames of context and no blending between chunks.
- The chunk boundaries can produce a one-frame brightness flash (for a 15 s video: around 6.8 s and 9.1 s). It becomes clearly visible when phase 2 starts at a low noise level, because phase 2 then keeps what the upscaler produced.
- The plugin makes the upscaler process the whole video in a single pass: up to 1024 latent frames, about 70 s, more than the 30 s maximum of MiniMax H3.
- VRAM is not an issue: in practice the upscaling peak stays about the level of phase 1 and well below phase 2, because the transformer is unloaded while the upscaler runs.

Both controls are hidden for other models, for One Phase, and for H3 variants that do not offer two phases (PDD, ControlNet-Union, Viggle, TTS).

## H3 Upscaler tab

A new **H3 Upscaler** tab runs MiniMax H3's phase 2 on a video that already exists: for instance a 512x512 video generated in one phase, whose proportions and motion you like, upscaled to 1024x1024 with phase 2 detail.

| Control | Default | Effect |
|---|---|---|
| **Video to upscale** | | The source video (compact preview). The panel next to it shows the source size, the output size, the number of frames processed and the number of sliding windows. |
| **Upscale Factor** | x2 | **x1 (refiner)**, x1.5, x2, x2.5 or x3. With x1, the output keeps the source size and phase 2 only refines the video (the latent upscaler is skipped). Other factors are applied to width and height (rounded to multiples of 32). The output is limited to **4096 pixels** on its longest side: beyond that, the factor is reduced and the panel shows the effective factor. |
| **MiniMax H3 Model (FL2VA or Ref2VA)** | FL2VA Pruned 20B | The model used for phase 2: FL2VA or Ref2VA (pruned or full, and their finetunes). |
| **References (Ref2VA)** | | Shown only for a Ref2VA model: **Reference Images** (same gallery as Media Generator) and up to three **Reference Videos**, used by phase 2 to keep identities and appearance. Name them in the Phase 2 Prompt (`<Picture 1>`, `<Video 1>`...) if the prompt describes them. |
| **Phase 2 Noise Level Start** | 0.6 | Same meaning as in the main form, with the live step count and sigmas. |
| **Phase 2 Steps** (WanGP 18+) | 2 | Number of phase 2 steps (0.6 with 2 steps = sigmas 0.6, 0.3, 0). |
| **Phase 2 Turbo LoRA Multiplier** | 0.5 | Strength of the phase 2 Turbo LoRA. |
| **Phase 2 Prompt** | `.` | `.` means no description: phase 2 only refines what it sees. **Copy the Prompt from the Video Properties** fills it with the prompt stored in a WanGP video's metadata. |
| **Seed** | -1 | -1 for random. |
| **Advanced** (closed by default) | | Three sub-tabs, as in Media Generator: |
| ↳ **LoRAs**: Apply the LoRAs selected in Media Generator | off | Uses the LoRAs and multipliers chosen in the **LoRAs** tab of **Media Generator** for the selected model, including what other LoRA plugins do there. A summary lists them with the multiplier phase 2 will use (`a;b` → `b`; Turbo LoRAs → 0). **Refresh from Media Generator** updates it. |
| ↳ **LoRAs**: MiniMax H3 RefMods (inline) | none | Shown only when the **MiniMax H3 RefMods** plugin is enabled and a Ref2VA model is selected. Same panel as in Media Generator (folder picker, image / video / audio RefMods with their strength, live reference counter), with its own selection: the RefMods picked here are used by this tab only, and the Media Generator selection is not used. |
| ↳ **Sliding Window**: Sliding Window Size / Windows Frames Overlap | 481 / 18 | Videos longer than the window size are processed in several windows (up to 737 frames, about 30 s). |
| ↳ **Misc.**: Text Encoder / Video VAE / DiT Denoising Priority | model default | Config variations of the selected model, as in Misc. The list follows the selected model. |
| ↳ **Misc.**: Override Frames Per Second | model default (24 fps) | Generation frame rate. **Source Video fps** uses the source frame rate (rounded). The source is resampled to this rate, and the frame limits apply at this rate. |
| ↳ **Misc.**: Override Memory Profile / Override Attention Mode | default | As in Misc. (the attention list includes Sol for H3). |
| ↳ **Misc.**: Phase 2 with Tiling | off | Same as **Two Phases with Tiling**: lower peak VRAM for large outputs (x2.5, x3), with a possible risk of seams. |

**Upscale** queues the task in WanGP's normal queue. A status line at the top of the tab shows the progress, then the path of the result. The result is saved in WanGP's output folder and appears in the Media Generator gallery, like any generation.

### How the upscaler works

The tab queues a normal two-phase generation for the selected FL2VA model, with these differences:
1. **Phase 1 is replaced by your video.** The video is resampled to 24 fps and encoded with the H3 VAE at the phase 1 resolution. During phase 1 (a single step), the plugin answers in place of the model with the exact prediction of that latent, so phase 1 ends exactly on your video without any model computation.
2. **The phase 1 to phase 2 scale is the chosen factor** (x1 to x3, limited to a 4096-pixel output) instead of WanGP's fixed x2. With x1 (refiner), phase 1 and phase 2 have the same size and the latent upscaler is skipped: the encoded video goes straight to phase 2. This only applies to H3 Upscaler tasks; Media Generator is unchanged. Phase 1 therefore runs at the source resolution, and the H3 latent upscaler enlarges by the chosen factor (single pass, see above).
3. **The soundtrack is kept.** The source audio is given to the model as its soundtrack (`Generate Video based on Soundtrack and Text Prompt (Soundtrack Kept)`: audio mode `A` for FL2VA, `AS` for Ref2VA), so it is not regenerated. A video without audio gets a silent track.
4. **Ref2VA references** (images, videos, RefMods) are given to the model as in Media Generator. The H3 pipeline re-encodes them at full resolution for phase 2, which is where they matter; in phase 1 they are encoded too but not used, since phase 1 is replaced by your video.
5. **Phase 2 runs as usual** at the target resolution, with the chosen noise level start, Turbo LoRA multiplier and prompt.

### Limits

- **Duration:** 107 frames minimum (4.5 s at 24 fps), 737 frames maximum (about 30 s at 24 fps). H3 lengths are 5 + 17n frames, so up to 16 frames at the end can be left out. Longer videos are cut at 737 frames. The panel shows how many frames are processed and how many sliding windows are used.
- **Sliding windows:** in each window, the pipeline keeps the overlap frames of the previous upscaled window as conditions and starts its target on the last overlap frame; the plugin replaces that window's phase 1 with the matching part of the source (the console shows the source frame range of each window). Continuity between windows relies on the overlap, as in any H3 sliding-window generation.
- **LoRAs:** Media Generator must have been used with the selected model at least once in the session for its LoRA selection to be known. Switching to another tab saves the Media Generator form, so the selection is up to date when you come to the H3 Upscaler tab.
- **Frame rate:** the output is 24 fps unless **Override Frames Per Second** is set. The source is resampled to the generation frame rate. MiniMax H3 was trained at 24 fps.
- **Large factors:** x2.5 and x3 ask the latent upscaler and phase 2 to invent much more detail than x2 (the upscaler was mainly used at x2), and phase 2 VRAM grows with the output area. Enable **Phase 2 with Tiling** if phase 2 runs out of VRAM.
- **Size:** the source is resized to multiples of 32 for phase 1. Odd sizes are slightly rescaled.
- **VRAM:** phase 2 at the target resolution needs as much VRAM as a normal phase 2 at that resolution.
- **Turbo finetunes:** a FL2VA finetune with a Turbo LoRA already merged gets the phase 2 Turbo LoRA on top. Lower the multiplier, or use the base FL2VA models.
- The console reports each step:
  ```
  [2 Phases Plus] H3 Upscaler: 512x512 -> 1024x1024, 362 frames, noise level start 0.6, Turbo multiplier 0.5, model minimax_h3_fl2va_pruned
  [2 Phases Plus] H3 Upscaler: phase 1 replaced by the source video '..._source.mp4', upscale x2
  [2 Phases Plus] H3 Upscaler: source encoded at 512x512, frames 0-361 at 24 fps, latent (1, 24, 107, 32, 32)
  ```
  The warning `the source video did not replace phase 1` means the result does not follow the source (see Troubleshooting).
- Temporary copies of the source and its soundtrack are written to the system temporary folder (`wan2gp_h3_upscaler`).

## Why

In H3 two-phase mode:
1. Phase 1 generates at half resolution with your settings and LoRAs.
2. A learned latent upscaler doubles the latent.
3. Phase 2 re-noises that latent to a high noise level (0.9035 by default) and denoises it in 3 Euler steps at the target resolution. It re-reads the whole prompt and uses the LightX2V FL2V 4-step v0.1 Turbo LoRA at 1.0.

At this noise level, phase 2 redraws much more than fine detail: volumes, body proportions, muscles and skin rendering are regenerated from the prompt. With motion or action LoRAs, the result can look exaggerated, even when the user LoRA is set to 0 for phase 2.
- The **multiplier** lowers the strength of the phase 2 Turbo LoRA.
- The **phase 2 prompt** lets you give phase 2 a lighter description, so it stops amplifying what the main prompt describes.
- A **lower noise level start** keeps more of the upscaled phase 1 result, so phase 2 has less freedom to redraw it.

## Installation

1. Copy the `wan2gp-2phases-plus` folder into WanGP's `plugins/` folder.
2. Start WanGP, open the **Plugins** tab, enable **2 Phases Plus** and save.
3. Restart WanGP.

The console should print:
```
Loaded plugin: 2 Phases Plus (from wan2gp-2phases-plus)
[2 Phases Plus] Phase 2 Noise Level Start minimum lowered to 0.01 (... loaded MiniMax H3 model definition(s) updated)
[2 Phases Plus] MiniMax H3 pipeline patched (generate, _encode_prompt, update_loras_slists, latent upscaler single pass)
[2 Phases Plus] hooked collect_custom_settings_from_inputs, prepare_inputs_dict
```

The controls are in the **General** tab of the advanced settings, below **Phase 2 Noise Level Start**.

## Usage

- **Phase 2 Turbo LoRA Multiplier**
  - The plugin's default is 0.5, the strength recommended by the official profile of that LoRA. 1.0 is WanGP's normal behaviour.
  - Phase 2 only has 3 steps, so very low values (close to 0) can leave the result soft or noisy.
- **Phase 2 Noise Level Start below 0.7**
  - WanGP's phase 2 schedule is `start, 0.6316, 0.3158, 0` (3 steps). The two intermediate values are fixed.
  - When the start is at or below 0.6316, the plugin keeps only the fixed values lower than the start, so the noise never goes back up:

    | Start | Phase 2 sigmas | Steps |
    |---|---|---|
    | 0.7 to 1.0 | start, 0.6316, 0.3158, 0 | 3 (WanGP) |
    | 0.32 to 0.6316 | start, 0.3158, 0 | 2 |
    | 0.01 to 0.3158 | start, 0 | 1 |

  - The lower the start, the closer the result stays to the upscaled phase 1 video: proportions and motion are better preserved, but phase 2 adds less fine detail. Very low values give almost the plain latent upscale.
  - The console reports the schedule used, for example `phase 2 noise level start 0.5 is at or below 0.6316: phase 2 sigmas 0.5, 0.3158, 0 (2 step(s))`.
- **Phase 2 Prompt**
  - Leave it empty to keep WanGP's normal behaviour.
  - `{prompt}` is replaced by the main prompt. For example, `{prompt} Natural body proportions, realistic anatomy.` appends text to the main prompt for phase 2 only.
  - For Ref2VA, keep the `<Picture N>` / `<Subject N>` tags of the main prompt if phase 2 should still refer to the references.
  - The same phase 2 prompt is used for every sliding window.

For every two-phase generation the console reports the values used:
```
[2 Phases Plus] values from the task settings: phase 2 Turbo LoRA multiplier 0.5, phase 2 prompt set (153 characters)
[2 Phases Plus] phase 2: Turbo LoRA 'minimax_h3_lightx2v_fl2v_turbo_4step_alpha16_v0.1.safetensors' multiplier set to 0.5 (default 1.0)
[2 Phases Plus] phase 2: using the Phase 2 Prompt (153 characters)
```

## How it works

- The values are stored in each task's `custom_settings`, under `two_phases_plus_turbo_multiplier` and `two_phases_plus_phase2_prompt`.
  - Queued tasks keep their own values.
  - The values are saved with **Save Settings** and written to the output file metadata.
  - Loading the settings of a video or image generated with them restores the controls.
- Two WanGP functions are wrapped, `prepare_inputs_dict` and `collect_custom_settings_from_inputs`, so the two values are not dropped as undeclared custom settings.
- The MiniMax H3 pipeline is patched at runtime. No WanGP file is modified.
  - Phase 2 is detected from the pipeline's own sequence: activation of the phase 2 LoRA multipliers, then phase 2 prompt encoding.
  - The multiplier replaces the phase 2 multiplier of the automatically added Turbo LoRA only. Your own LoRAs keep the multipliers you set with the `phase1;phase2` syntax.
  - The phase 2 prompt replaces the main prompt for the phase 2 encoding only. Phase 1 and the optional Audio Refinement Extra Phase keep the main prompt.
  - The audio is generated in phase 1 and frozen in phase 2, so the phase 2 prompt only affects the video.
  - Latent upscaler: `MiniMaxH3LatentUpscaler.forward` is wrapped and the upscaler's `temporal_chunk_size` is raised to 1024 before each two-phase generation. The console reports it once: `latent upscaler: temporal chunking disabled (single pass up to 1024 latent frames, WanGP default 16)`.
  - Noise level range: the minimum of `switch_threshold` is lowered in the MiniMax H3 model definitions, and in the H3 handler so that rebuilt definitions keep it. WanGP then accepts the lower values both in the slider and in its settings validation. For a start at or below 0.6316, the phase 2 schedule (`H3_PHASE_2_SIGMAS`) is replaced for that generation only, then restored.

## Troubleshooting

- **No `[2 Phases Plus] values ...` line during a two-phase generation**: the pipeline patch is not active. Change any of the two controls once (this reinstalls the patch) and generate again. If the line is still missing, another plugin replaces the MiniMax H3 pipeline in a way this plugin cannot follow: disable the other H3 plugins one by one to find it.
- **`values from the UI controls` instead of `values from the task settings`**: another plugin replaced WanGP's settings functions, so the values are not stored in the task. The plugin then uses the current values of the controls, which works for direct generations but not for queued tasks with different values, and the values are not written to the metadata. Changing a control once reinstalls the hooks (`re-hooked ...` in the console).
- Lines `re-hooked ...` or `re-patched ...` are normal: they mean another plugin replaced a function after this plugin, and the plugin installed itself again on top.

- **H3 Upscaler warning `the source video did not replace phase 1`**: something replaced the H3 transformer's `forward` between the start of the generation and phase 1, so phase 1 was a real one-step generation. Disable the other H3 plugins to find the conflict, and report it.

## Notes and limits

- The control values are shared by the whole session: they are kept when you switch between H3 models.
- When a queued task is edited, the current values of the controls are applied to it.
- If a future WanGP version changes the H3 two-phase code, the plugin may stop applying its values. It never blocks generation: the console simply no longer reports the phase 2 lines.
- Settings saved with a noise level start below 0.7 need this plugin: without it, WanGP rejects them (minimum 0.7).
- Tested against WanGP v17.17 and v18.02. WanGP 18 added a native **Phase 2 Steps** setting (default 3) and accelerator LoRAs given before `|` that replace the automatic Turbo; the plugin follows both:
  - the Phase 2 Turbo LoRA Multiplier only applies to WanGP's automatic Turbo LoRA; when your own accelerator LoRAs replace it (nonzero phase 2 multiplier before `|`), the plugin leaves them as you set them;
  - with 3 Phase 2 Steps, WanGP's fixed schedule (start, 0.6316, 0.3158, 0) is kept above 0.6316; at or below, where it would go back up in noise, the plugin makes WanGP use evenly spaced sigmas from the start (for example 0.6, 0.4, 0.2, 0). Other step counts are already evenly spaced by WanGP. The live indicator and the H3 Upscaler tab (which gets its own **Phase 2 Steps** slider on WanGP 18+) show the real schedule;
  - on older WanGP versions, the noise level start still decides the step count (3, 2 or 1 step), as before.

## Uninstall

Disable the plugin in the **Plugins** tab and restart WanGP, or delete the `plugins/wan2gp-2phases-plus` folder.
