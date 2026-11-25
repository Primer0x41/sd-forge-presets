import hashlib
import json
import logging
import re
from pathlib import Path

import gradio as gr

from modules import script_callbacks, sd_models, shared, ui_common, ui_components
from modules_forge import main_entry
from modules_forge.presets import PresetArch, is_video, use_distill, use_shift

logger = logging.getLogger(__name__)

PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"

SCOPE_LABELS = [
    ("Checkpoint & VAE", "model"),
    ("Generation Settings", "gen"),
    ("Hires. fix", "hires"),
    ("Refiner", "refiner"),
    ("Prompts", "prompts"),
    ("Seed", "seed"),
]
DEFAULT_SCOPES = ["model", "gen", "hires", "refiner"]

# (scope, tab_or_none, key, elem_id)
FIELD_SPECS: list[tuple[str, str | None, str, str]] = [
    # Model & Backend (global)
    ("model", None, "ui_preset", "forge_ui_preset"),
    ("model", None, "checkpoint", "setting_sd_model_checkpoint"),
    ("model", None, "vae_te", "setting_sd_modules"),
    ("model", None, "low_bits", "forge_ui_dtype"),
    # txt2img Generation
    ("gen", "txt2img", "sampling", "txt2img_sampling"),
    ("gen", "txt2img", "scheduler", "txt2img_scheduler"),
    ("gen", "txt2img", "steps", "txt2img_steps"),
    ("gen", "txt2img", "width", "txt2img_width"),
    ("gen", "txt2img", "height", "txt2img_height"),
    ("gen", "txt2img", "batch_count", "txt2img_batch_count"),
    ("gen", "txt2img", "batch_size", "txt2img_batch_size"),
    ("gen", "txt2img", "distilled_cfg_scale", "txt2img_distilled_cfg_scale"),
    ("gen", "txt2img", "cfg_scale", "txt2img_cfg_scale"),
    ("gen", "txt2img", "rescale_cfg_scale", "txt2img_rescale_cfg_scale"),
    ("gen", "txt2img", "enable_mahiro", "txt2img_enable_mahiro"),
    ("gen", "txt2img", "override_settings", "txt2img_override_settings"),
    # Hires. fix (txt2img) - InputAccordion value lives on *-checkbox
    ("hires", "txt2img", "enable_hr", "txt2img_hr-checkbox"),
    ("hires", "txt2img", "hr_upscaler", "txt2img_hr_upscaler"),
    ("hires", "txt2img", "hr_steps", "txt2img_hires_steps"),
    ("hires", "txt2img", "denoising_strength", "txt2img_denoising_strength"),
    ("hires", "txt2img", "hr_scale", "txt2img_hr_scale"),
    ("hires", "txt2img", "hr_resize_x", "txt2img_hr_resize_x"),
    ("hires", "txt2img", "hr_resize_y", "txt2img_hr_resize_y"),
    ("hires", "txt2img", "hr_distilled_cfg", "txt2img_hr_distilled_cfg"),
    ("hires", "txt2img", "hr_cfg", "txt2img_hr_cfg"),
    ("hires", "txt2img", "hr_checkpoint", "hr_checkpoint"),
    ("hires", "txt2img", "hr_vae_te", "hr_vae_te"),
    ("hires", "txt2img", "hr_sampler", "hr_sampler"),
    ("hires", "txt2img", "hr_scheduler", "hr_scheduler"),
    # img2img Generation
    ("gen", "img2img", "sampling", "img2img_sampling"),
    ("gen", "img2img", "scheduler", "img2img_scheduler"),
    ("gen", "img2img", "steps", "img2img_steps"),
    ("gen", "img2img", "resize_mode", "resize_mode"),
    ("gen", "img2img", "width", "img2img_width"),
    ("gen", "img2img", "height", "img2img_height"),
    ("gen", "img2img", "scale", "img2img_scale"),
    ("gen", "img2img", "batch_count", "img2img_batch_count"),
    ("gen", "img2img", "batch_size", "img2img_batch_size"),
    ("gen", "img2img", "denoising_strength", "img2img_denoising_strength"),
    ("gen", "img2img", "distilled_cfg_scale", "img2img_distilled_cfg_scale"),
    ("gen", "img2img", "cfg_scale", "img2img_cfg_scale"),
    ("gen", "img2img", "image_cfg_scale", "img2img_image_cfg_scale"),
    ("gen", "img2img", "rescale_cfg_scale", "img2img_rescale_cfg_scale"),
    ("gen", "img2img", "enable_mahiro", "img2img_enable_mahiro"),
    ("gen", "img2img", "mask_blur", "img2img_mask_blur"),
    ("gen", "img2img", "mask_alpha", "img2img_mask_alpha"),
    ("gen", "img2img", "mask_mode", "img2img_mask_mode"),
    ("gen", "img2img", "inpainting_fill", "img2img_inpainting_fill"),
    ("gen", "img2img", "inpaint_full_res", "img2img_inpaint_full_res"),
    ("gen", "img2img", "inpaint_full_res_padding", "img2img_inpaint_full_res_padding"),
    ("gen", "img2img", "soft_inpainting_enabled", "soft_inpainting_enabled-checkbox"),
    ("gen", "img2img", "mask_blend_power", "mask_blend_power"),
    ("gen", "img2img", "mask_blend_scale", "mask_blend_scale"),
    ("gen", "img2img", "inpaint_detail_preservation", "inpaint_detail_preservation"),
    ("gen", "img2img", "composite_mask_influence", "composite_mask_influence"),
    ("gen", "img2img", "composite_difference_threshold", "composite_difference_threshold"),
    ("gen", "img2img", "composite_difference_contrast", "composite_difference_contrast"),
    ("gen", "img2img", "override_settings", "img2img_override_settings"),
    # Refiner (per tab) - ScriptBuiltinUI omits script_/refiner_ prefix; InputAccordion lives on *-checkbox
    ("refiner", "txt2img", "refiner_enable", "txt2img_enable-checkbox"),
    ("refiner", "txt2img", "refiner_checkpoint", "txt2img_checkpoint"),
    ("refiner", "txt2img", "refiner_switch_at", "txt2img_switch_at"),
    ("refiner", "txt2img", "refiner_cfg", "txt2img_cfg"),
    ("refiner", "img2img", "refiner_enable", "img2img_enable-checkbox"),
    ("refiner", "img2img", "refiner_checkpoint", "img2img_checkpoint"),
    ("refiner", "img2img", "refiner_switch_at", "img2img_switch_at"),
    ("refiner", "img2img", "refiner_cfg", "img2img_cfg"),
    # Prompts (per tab)
    ("prompts", "txt2img", "prompt", "txt2img_prompt"),
    ("prompts", "txt2img", "neg_prompt", "txt2img_neg_prompt"),
    ("prompts", "txt2img", "styles", "txt2img_styles"),
    ("prompts", "txt2img", "hires_prompt", "hires_prompt"),
    ("prompts", "txt2img", "hires_neg_prompt", "hires_neg_prompt"),
    ("prompts", "img2img", "prompt", "img2img_prompt"),
    ("prompts", "img2img", "neg_prompt", "img2img_neg_prompt"),
    ("prompts", "img2img", "styles", "img2img_styles"),
    # Seed (per tab)
    ("seed", "txt2img", "seed", "txt2img_seed"),
    ("seed", "txt2img", "subseed_show", "txt2img_subseed_show"),
    ("seed", "txt2img", "subseed", "txt2img_subseed"),
    ("seed", "txt2img", "subseed_strength", "txt2img_subseed_strength"),
    ("seed", "txt2img", "seed_resize_from_w", "txt2img_seed_resize_from_w"),
    ("seed", "txt2img", "seed_resize_from_h", "txt2img_seed_resize_from_h"),
    ("seed", "img2img", "seed", "img2img_seed"),
    ("seed", "img2img", "subseed_show", "img2img_subseed_show"),
    ("seed", "img2img", "subseed", "img2img_subseed"),
    ("seed", "img2img", "subseed_strength", "img2img_subseed_strength"),
    ("seed", "img2img", "seed_resize_from_w", "img2img_seed_resize_from_w"),
    ("seed", "img2img", "seed_resize_from_h", "img2img_seed_resize_from_h"),
]

_TRACKED_ELEM_IDS = {elem_id for _, _, _, elem_id in FIELD_SPECS}

# Maps main_entry.on_preset_change() 22-element output list to our elem_ids
_MAIN_ENTRY_OUTPUT_ELEM_IDS = [
    "setting_sd_model_checkpoint",
    "setting_sd_modules",
    "forge_ui_dtype",
    "txt2img_steps",
    "txt2img_hires_steps",
    "img2img_steps",
    "txt2img_sampling",
    "img2img_sampling",
    "txt2img_scheduler",
    "img2img_scheduler",
    "txt2img_width",
    "img2img_width",
    "txt2img_height",
    "img2img_height",
    "txt2img_cfg_scale",
    "txt2img_hr_cfg",
    "img2img_cfg_scale",
    "txt2img_distilled_cfg_scale",
    "txt2img_hr_distilled_cfg",
    "img2img_distilled_cfg_scale",
    "txt2img_batch_size",
    "img2img_batch_size",
]


def _static_defaults() -> dict:
    return {
        "txt2img_width": 1024,
        "txt2img_height": 1024,
        "txt2img_batch_count": 1,
        "txt2img_rescale_cfg_scale": 0.0,
        "txt2img_enable_mahiro": False,
        "txt2img_override_settings": [],
        "txt2img_hr-checkbox": False,
        "txt2img_hr_upscaler": getattr(shared, "latent_upscale_default_mode", "Latent"),
        "txt2img_denoising_strength": 0.6,
        "txt2img_hr_scale": 2.0,
        "txt2img_hr_resize_x": 0,
        "txt2img_hr_resize_y": 0,
        "hr_checkpoint": "Use same checkpoint",
        "hr_vae_te": ["Use same choices"],
        "hr_sampler": "Use same sampler",
        "hr_scheduler": "Use same scheduler",
        "resize_mode": 1,
        "img2img_width": 1024,
        "img2img_height": 1024,
        "img2img_scale": 1.0,
        "img2img_batch_count": 1,
        "img2img_denoising_strength": 0.6,
        "img2img_image_cfg_scale": 1.5,
        "img2img_rescale_cfg_scale": 0.0,
        "img2img_enable_mahiro": False,
        "img2img_mask_blur": 4,
        "img2img_mask_alpha": 0,
        "img2img_mask_mode": 0,
        "img2img_inpainting_fill": 1,
        "img2img_inpaint_full_res": 1,
        "img2img_inpaint_full_res_padding": 32,
        "soft_inpainting_enabled-checkbox": False,
        "mask_blend_power": 1,
        "mask_blend_scale": 0.5,
        "inpaint_detail_preservation": 4,
        "composite_mask_influence": 0,
        "composite_difference_threshold": 0.5,
        "composite_difference_contrast": 2,
        "img2img_override_settings": [],
        "txt2img_enable-checkbox": False,
        "txt2img_checkpoint": "None",
        "txt2img_switch_at": 0.875,
        "txt2img_cfg": 0.0,
        "img2img_enable-checkbox": False,
        "img2img_checkpoint": "None",
        "img2img_switch_at": 0.875,
        "img2img_cfg": 0.0,
        "txt2img_prompt": "",
        "txt2img_neg_prompt": "",
        "txt2img_styles": [],
        "hires_prompt": "",
        "hires_neg_prompt": "",
        "img2img_prompt": "",
        "img2img_neg_prompt": "",
        "img2img_styles": [],
        "txt2img_seed": -1,
        "txt2img_subseed_show": False,
        "txt2img_subseed": -1,
        "txt2img_subseed_strength": 0.0,
        "txt2img_seed_resize_from_w": 0,
        "txt2img_seed_resize_from_h": 0,
        "img2img_seed": -1,
        "img2img_subseed_show": False,
        "img2img_subseed": -1,
        "img2img_subseed_strength": 0.0,
        "img2img_seed_resize_from_w": 0,
        "img2img_seed_resize_from_h": 0,
    }


def _preset_slug(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip()).strip("._-")
    if not clean or clean in {".", ".."}:
        raise ValueError("Preset name must contain at least one alphanumeric character.")
    return clean


def _iter_preset_files() -> list[tuple[Path, dict]]:
    if not PRESETS_DIR.is_dir():
        return []
    entries: list[tuple[Path, dict]] = []
    for file in sorted(PRESETS_DIR.glob("*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                entries.append((file, data))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read preset %s: %s", file, exc)
    return entries


def _preset_path(name: str) -> Path:
    clean_name = name.strip()
    presets_root = PRESETS_DIR.resolve()

    for file, data in _iter_preset_files():
        if data.get("name") == clean_name:
            resolved = file.resolve()
            if resolved.parent == presets_root:
                return resolved

    slug = _preset_slug(clean_name)
    candidate = (PRESETS_DIR / f"{slug}.json").resolve()
    if candidate.parent != presets_root:
        raise ValueError("Invalid preset path.")

    if candidate.exists():
        suffix = hashlib.sha1(clean_name.encode("utf-8")).hexdigest()[:6]
        candidate = (PRESETS_DIR / f"{slug}_{suffix}.json").resolve()
        if candidate.parent != presets_root:
            raise ValueError("Invalid preset path.")

    return candidate


def list_presets(tab: str | None = None) -> list[str]:
    names = []
    for file, data in _iter_preset_files():
        if tab is not None and (data.get("tab") or "txt2img") != tab:
            continue
        names.append(str(data.get("name") or file.stem))
    return sorted(set(names), key=str.casefold)


def load_preset(name: str) -> dict | None:
    if not name or name == "None":
        return None
    entries = _iter_preset_files()
    for _file, data in entries:
        if data.get("name") == name:
            return data
    for file, data in entries:
        if file.stem == name:
            return data
    return None


def save_preset(name: str, payload: dict) -> Path:
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    clean_name = name.strip()
    data = {"version": 1, "name": clean_name, **payload}
    path = _preset_path(clean_name)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def delete_preset(name: str) -> bool:
    if not name or name == "None":
        return False
    entries = _iter_preset_files()
    target = next((file for file, data in entries if data.get("name") == name), None)
    if target is None:
        target = next((file for file, _data in entries if file.stem == name), None)
    if target is None:
        return False
    try:
        target.unlink()
        return True
    except OSError as exc:
        logger.warning("Failed to delete preset %s: %s", target, exc)
        return False


class _SdForgePresetsState:
    def __init__(self):
        self.components: dict[str, gr.components.Component] = {}
        self.ui_dropdown: gr.Dropdown | None = None
        self.ui_manage: ui_components.ToolButton | None = None
        self.ui_delete: ui_components.ToolButton | None = None
        self.ui_apply_rest: gr.Button | None = None
        self.ui_sync_tab: gr.Button | None = None
        self.modal_name: gr.Textbox | None = None
        self.modal_scopes: gr.CheckboxGroup | None = None
        self.modal_tab: gr.Textbox | None = None
        self.modal_dialog: gr.Group | None = None
        self.modal_save: gr.Button | None = None
        self.modal_close: gr.Button | None = None
        self.selected_by_tab: dict[str, str] = {"txt2img": "None", "img2img": "None"}
        self._wired = False

    def reset(self):
        self.components.clear()
        self.ui_dropdown = None
        self.ui_manage = None
        self.ui_delete = None
        self.ui_apply_rest = None
        self.ui_sync_tab = None
        self.modal_name = None
        self.modal_scopes = None
        self.modal_tab = None
        self.modal_dialog = None
        self.modal_save = None
        self.modal_close = None
        self.selected_by_tab = {"txt2img": "None", "img2img": "None"}
        self._wired = False


_state = _SdForgePresetsState()


def _build_field_updates(
    data: dict,
    specs: list[tuple[str, str | None, str, str]],
    *,
    skip_ui_preset: bool = False,
    skip_checkpoint: bool = False,
):
    scopes = set(data.get("scopes", []))
    preset_tab = data.get("tab", "txt2img")
    ui_preset = data.get("model", {}).get("ui_preset")

    d_args = {}
    batch_args = {}
    if "model" in scopes and ui_preset:
        if use_shift(ui_preset):
            d_args = {"visible": getattr(shared.opts, f"{ui_preset}_show_shift", True), "label": "Shift"}
        elif use_distill(ui_preset):
            d_args = {"visible": True, "label": "Distilled CFG Scale"}
        else:
            d_args = {"visible": False}

        if (fps := is_video(ui_preset)) > 1:
            batch_args = {"minimum": 1, "maximum": fps * 15 + 1, "step": fps, "label": "Frames"}
        else:
            batch_args = {"minimum": 1, "maximum": 8, "step": 1, "label": "Batch Size"}

    updates = []
    for scope, tab, key, _elem_id in specs:
        if scope not in scopes or (tab is not None and tab != preset_tab):
            updates.append(gr.skip())
            continue
        if skip_ui_preset and scope == "model" and key == "ui_preset":
            updates.append(gr.skip())
            continue
        if skip_checkpoint and scope == "model" and key == "checkpoint":
            updates.append(gr.skip())
            continue
        section = data.get(scope, {})
        if key not in section:
            updates.append(gr.skip())
            continue
        val = section[key]
        if key == "override_settings" and val is None:
            val = []
        if key in ("distilled_cfg_scale", "hr_distilled_cfg") and d_args:
            updates.append(gr.update(value=val, **d_args))
        elif key == "batch_size" and batch_args:
            updates.append(gr.update(value=val, **batch_args))
        else:
            updates.append(gr.update(value=val))
    return updates


def _open_modal_state(cur: str) -> tuple[str, list[str]]:
    if not cur or cur == "None":
        return "", DEFAULT_SCOPES
    data = load_preset(cur)
    scopes = data.get("scopes") if isinstance(data, dict) else None
    return cur, (scopes if scopes else DEFAULT_SCOPES)


def _build_toolbar():
    _state.ui_dropdown = gr.Dropdown(
        label="Preset",
        choices=["None"] + list_presets("txt2img"),
        value="None",
        elem_id="sd_forge_presets_dropdown",
    )
    _state.ui_manage = ui_components.ToolButton(
        value="💾",
        elem_id="sd_forge_presets_manage",
        tooltip="Save / Manage presets",
    )
    _state.ui_delete = ui_components.ToolButton(
        value="🗑️",
        elem_id="sd_forge_presets_delete",
        tooltip="Delete selected preset",
    )
    _state.ui_apply_rest = gr.Button(visible=False, elem_id="sd_forge_presets_apply_rest")
    _state.ui_sync_tab = gr.Button(visible=False, elem_id="sd_forge_presets_sync_tab")

    with gr.Group(elem_id="sd_forge_presets_dialog", elem_classes="popup-dialog") as dialog:
        _state.modal_tab = gr.Textbox(value="txt2img", visible=False, elem_id="sd_forge_presets_modal_tab")
        _state.modal_name = gr.Textbox(
            label="Preset Name",
            value="",
            placeholder="Enter preset name (existing name overwrites)...",
            elem_id="sd_forge_presets_modal_name",
        )
        _state.modal_scopes = gr.CheckboxGroup(
            label="Sections to Save",
            show_label=False,
            choices=SCOPE_LABELS,
            value=DEFAULT_SCOPES,
            elem_id="sd_forge_presets_modal_scopes",
        )
        with gr.Row(elem_id="sd_forge_presets_modal_actions"):
            _state.modal_save = gr.Button("Save", variant="primary", elem_id="sd_forge_presets_modal_save")
            _state.modal_close = gr.Button("Close", variant="secondary", elem_id="sd_forge_presets_modal_close")

    _state.modal_dialog = dialog


def _wire_events():
    if _state._wired or _state.ui_dropdown is None:
        return
    _state._wired = True

    if _state.modal_dialog is not None:
        ui_common.setup_dialog(button_show=_state.ui_manage, dialog=_state.modal_dialog, button_close=_state.modal_close)

    active_specs = [(scope, tab, key, elem_id) for scope, tab, key, elem_id in FIELD_SPECS if elem_id in _state.components]
    target_comps = [_state.components[elem_id] for _, _, _, elem_id in active_specs]
    non_model_specs = [(scope, tab, key, elem_id) for scope, tab, key, elem_id in active_specs if scope != "model"]
    non_model_comps = [_state.components[elem_id] for _, _, _, elem_id in non_model_specs]

    def _restore_defaults(active_tab: str = "txt2img"):
        tab = "img2img" if active_tab == "img2img" else "txt2img"
        preset = getattr(shared.opts, "forge_preset", "sd")
        raw_updates = main_entry.on_preset_change(preset)
        assert len(raw_updates) == len(_MAIN_ENTRY_OUTPUT_ELEM_IDS), (
            f"main_entry.on_preset_change() returned {len(raw_updates)} updates, expected {len(_MAIN_ENTRY_OUTPUT_ELEM_IDS)}"
        )
        default_map = dict(zip(_MAIN_ENTRY_OUTPUT_ELEM_IDS, raw_updates))
        main_entry._load_presets(
            default_map["setting_sd_model_checkpoint"].get("value"),
            default_map["setting_sd_modules"].get("value", []),
            default_map["forge_ui_dtype"].get("value", "Automatic"),
            preset,
        )
        static = _static_defaults()
        updates = []
        for _scope, comp_tab, _key, elem_id in active_specs:
            if comp_tab is not None and comp_tab != tab:
                updates.append(gr.skip())
                continue
            upd = default_map.get(elem_id)
            if isinstance(upd, dict) and "value" in upd:
                updates.append(upd)
            elif elem_id in static:
                updates.append(gr.update(value=static[elem_id]))
            else:
                updates.append(gr.skip())
        return updates

    def apply_preset(preset_name: str, active_tab: str = "txt2img"):
        tab = "img2img" if active_tab == "img2img" else "txt2img"
        if not preset_name or preset_name == "None":
            _state.selected_by_tab[tab] = "None"
            return _restore_defaults(tab)

        data = load_preset(preset_name)
        if not data:
            return [gr.skip()] * len(target_comps)

        _state.selected_by_tab[tab] = preset_name
        scopes = set(data.get("scopes", []))
        model_data = data.get("model", {})
        ui_preset = model_data.get("ui_preset")
        skip_ui_preset = False
        skip_checkpoint = False

        if "model" in scopes and ui_preset:
            disabled_presets = set(getattr(getattr(shared, "cmd_opts", None), "disable_preset", None) or [])
            valid_presets = set(PresetArch.choices()) - disabled_presets
            if ui_preset not in valid_presets:
                gr.Warning(f"UI preset '{ui_preset}' is not available.")
                active_preset = getattr(shared.opts, "forge_preset", "sd")
                skip_ui_preset = True
            else:
                shared.opts.set("forge_preset", ui_preset)
                active_preset = ui_preset

            needs_refresh = False
            if "low_bits" in model_data:
                new_dtype = model_data["low_bits"]
                if new_dtype != getattr(shared.opts, "forge_unet_storage_dtype", None):
                    needs_refresh = True
                main_entry.dtype_change(new_dtype, active_preset, save=False, refresh=False)
            if "vae_te" in model_data:
                if main_entry.modules_change(model_data["vae_te"], active_preset, save=False, refresh=False):
                    needs_refresh = True
            if "checkpoint" in model_data:
                ckpt_name = model_data["checkpoint"]
                if ckpt_name and sd_models.get_closet_checkpoint_match(ckpt_name) is None:
                    gr.Warning(f"Checkpoint '{ckpt_name}' not found.")
                    skip_checkpoint = True
                    if needs_refresh:
                        main_entry.refresh_model_loading_parameters(refresh=True)
                    shared.opts.save(shared.config_filename)
                else:
                    changed = main_entry.checkpoint_change(ckpt_name, active_preset, save=True, refresh=True)
                    if not changed:
                        if needs_refresh:
                            main_entry.refresh_model_loading_parameters(refresh=True)
                        shared.opts.save(shared.config_filename)
            else:
                if needs_refresh:
                    main_entry.refresh_model_loading_parameters(refresh=True)
                shared.opts.save(shared.config_filename)

        return _build_field_updates(data, active_specs, skip_ui_preset=skip_ui_preset, skip_checkpoint=skip_checkpoint)

    def apply_rest(preset_name: str):
        data = load_preset(preset_name)
        if not data:
            return [gr.skip()] * len(non_model_comps)
        return _build_field_updates(data, non_model_specs)

    def refresh_dropdown(selected: str | None = None, tab: str = "txt2img"):
        norm_tab = "img2img" if tab == "img2img" else "txt2img"
        presets = list_presets(norm_tab)
        dd_val = selected if selected in presets else "None"
        _state.selected_by_tab[norm_tab] = dd_val
        return gr.update(choices=["None"] + presets, value=dd_val)

    def on_tab_change(active_tab: str):
        tab = "img2img" if active_tab == "img2img" else "txt2img"
        return refresh_dropdown(_state.selected_by_tab.get(tab, "None"), tab)

    def on_save(name: str, scopes: list[str], active_tab: str, *values):
        name = (name or "").strip()
        if not name or name == "None":
            gr.Warning("Enter a valid preset name.")
            return gr.skip()
        if not scopes:
            gr.Warning("Select at least one section to save.")
            return gr.skip()

        tab = "img2img" if active_tab == "img2img" else "txt2img"
        scope_set = set(scopes)
        payload: dict = {
            "tab": tab,
            "scopes": [s for _, s in SCOPE_LABELS if s in scope_set],
        }
        for (scope, comp_tab, key, _), val in zip(active_specs, values):
            if scope in scope_set and (comp_tab is None or comp_tab == tab):
                if key == "override_settings" and val is None:
                    val = []
                payload.setdefault(scope, {})[key] = val

        try:
            save_preset(name, payload)
        except ValueError as exc:
            gr.Warning(str(exc))
            return gr.skip()

        gr.Info(f"Saved {tab} preset '{name}'.")
        return refresh_dropdown(name, tab)

    def on_delete(name: str, active_tab: str = "txt2img"):
        tab = "img2img" if active_tab == "img2img" else "txt2img"
        name = (name or "").strip()
        if not name or name == "None":
            return gr.skip(), *([gr.skip()] * len(target_comps))
        if delete_preset(name):
            gr.Info(f"Deleted preset '{name}'.")
            return refresh_dropdown("None", tab), *_restore_defaults(tab)
        return refresh_dropdown("None", tab), *([gr.skip()] * len(target_comps))

    # Wait for Forge's ui_forge_preset.change -> clickLoraRefresh chain when UI Preset changes
    # so architecture defaults cannot overwrite saved steps/cfg/size.
    _js_prepare_apply = """(name, _tab) => {
        const el = gradioApp().querySelector('#forge_ui_preset input, #forge_ui_preset select');
        window.__sdForgePresetsPrevArch = el ? el.value : '';
        window.__sdForgePresetsPendingRest = Boolean(name && name !== 'None');
        if (window.__sdForgePresetsTimer) {
            clearTimeout(window.__sdForgePresetsTimer);
            window.__sdForgePresetsTimer = null;
        }
        if (typeof window.clickLoraRefresh === 'function' && !window.__sdForgePresetsWrappedLora) {
            const orig = window.clickLoraRefresh;
            window.clickLoraRefresh = function(...args) {
                const res = orig.apply(this, args);
                if (window.__sdForgePresetsPendingRest) {
                    window.__sdForgePresetsPendingRest = false;
                    if (window.__sdForgePresetsTimer) {
                        clearTimeout(window.__sdForgePresetsTimer);
                        window.__sdForgePresetsTimer = null;
                    }
                    gradioApp().querySelector('#sd_forge_presets_apply_rest')?.click();
                }
                return res;
            };
            window.__sdForgePresetsWrappedLora = true;
        }
        const tab = typeof window.sdForgePresetsActiveTab === 'function' ? window.sdForgePresetsActiveTab() : 'txt2img';
        return [name, tab];
    }"""

    _js_finish_apply = """() => {
        const el = gradioApp().querySelector('#forge_ui_preset input, #forge_ui_preset select');
        const cur = el ? el.value : '';
        if (!window.__sdForgePresetsPendingRest || cur === window.__sdForgePresetsPrevArch) {
            window.__sdForgePresetsPendingRest = false;
            return;
        }
        window.__sdForgePresetsTimer = setTimeout(() => {
            if (window.__sdForgePresetsPendingRest) {
                window.__sdForgePresetsPendingRest = false;
                gradioApp().querySelector('#sd_forge_presets_apply_rest')?.click();
            }
        }, 1500);
    }"""

    _state.ui_dropdown.input(
        fn=apply_preset,
        _js=_js_prepare_apply,
        inputs=[_state.ui_dropdown, _state.modal_tab],
        outputs=target_comps,
        queue=False,
        show_progress=False,
    ).then(
        fn=None,
        _js=_js_finish_apply,
        queue=False,
        show_progress=False,
    )
    _state.ui_apply_rest.click(fn=apply_rest, inputs=[_state.ui_dropdown], outputs=non_model_comps, queue=False, show_progress=False)
    if _state.ui_sync_tab is not None:
        _state.ui_sync_tab.click(
            fn=on_tab_change,
            _js="(_tab) => [typeof window.sdForgePresetsActiveTab === 'function' ? window.sdForgePresetsActiveTab() : 'txt2img']",
            inputs=[_state.modal_tab],
            outputs=[_state.ui_dropdown],
            queue=False,
            show_progress=False,
        )

    _state.ui_manage.click(fn=_open_modal_state, inputs=[_state.ui_dropdown], outputs=[_state.modal_name, _state.modal_scopes], queue=False, show_progress=False)
    _state.ui_delete.click(
        fn=on_delete,
        _js='(name, _tab) => { const tab = typeof window.sdForgePresetsActiveTab === "function" ? window.sdForgePresetsActiveTab() : "txt2img"; if(!name || name === "None") return ["None", tab]; return [confirm("Delete preset " + name + "?") ? name : "None", tab]; }',
        inputs=[_state.ui_dropdown, _state.modal_tab],
        outputs=[_state.ui_dropdown, *target_comps],
        queue=False,
        show_progress=False,
    )

    _state.modal_save.click(
        fn=on_save,
        _js="(name, scopes, _tab, ...rest) => [name, scopes, (typeof window.sdForgePresetsActiveTab === 'function' ? window.sdForgePresetsActiveTab() : 'txt2img'), ...rest]",
        inputs=[_state.modal_name, _state.modal_scopes, _state.modal_tab, *target_comps],
        outputs=[_state.ui_dropdown],
        queue=False,
        show_progress=False,
    ).success(fn=None, js="closePopup", queue=False, show_progress=False)


def _on_after_component(component, **_kwargs):
    elem_id = getattr(component, "elem_id", None)
    if elem_id in _TRACKED_ELEM_IDS:
        _state.components[elem_id] = component
    if elem_id == "forge_ui_dtype":
        _build_toolbar()
    elif elem_id == "footer":
        _wire_events()


script_callbacks.on_before_ui(_state.reset)
script_callbacks.on_after_component(_on_after_component)
