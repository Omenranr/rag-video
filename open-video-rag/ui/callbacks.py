import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional
import shutil
import time
from datetime import datetime
import json
import os
import logging
from logging.handlers import RotatingFileHandler

from resources.consts import *
from utils.visual import list_videos, find_existing_outputs_for_video, has_visual_artifacts, build_visual_index, ctx_md_from_hits_aggregated, hits_from_scene_rows, load_scene_rows_from_csv
from utils.stt import find_any_srt, find_audio_file_in, build_index_from_srt, srt_to_seconds
from utils.common import find_existing_indexes, safe_stem, load_index, format_source_label, html_escape
from utils.chat import sanitize_scene_output, trim_history_messages, contextualize_question, format_history_as_text, redact_cfg_for_preview, messages_preview_text
from core.intent_detection import detect_entity_intent, llm_decide_search, wants_recent_info, wants_web_search_explicit, llm_detect_intent_entities
from core.entity_matching import llm_match_entities_to_keys, fuzzy_pick_entity
from core.chat_context import compile_context_blocks_multi, auto_select_sources_from_query, load_entities_scenes_from_context
from core.extraction import run_extractor, run_visual_strategy, run_parallel_pipelines
from chat.llm import llm_chat_stream
from tools.exa import exa_search_with_contents


# ---------- Logging setup ----------
LOG_NAME = "video_rag"
LOG_PATH = os.getenv("APP_LOG_PATH", "app.log")
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()

logger = logging.getLogger(LOG_NAME)

if not logger.handlers:
    logger.setLevel(LOG_LEVEL)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)


def _log_step(step: str, **kwargs):
    """Structured log for milestones. Redacts obvious secrets."""
    try:
        safe = {}
        for k, v in (kwargs or {}).items():
            if k.lower() in {"api_key", "oa_api_key", "an_api_key", "exa_api_key"}:
                safe[k] = "***redacted***"
            else:
                safe[k] = v
        logger.info("%s | %s", step, json.dumps(safe, ensure_ascii=False))
    except Exception:
        # Never crash on logging
        logger.info("%s | %s", step, str(kwargs))


# ---------------------
# Gradio callbacks
# ---------------------
def do_scan(folder):
    _log_step("do_scan:start", folder=folder)
    vids = list_videos(folder)
    _log_step("do_scan:found_videos", count=len(vids))
    dd_update = gr.update(choices=vids, value=(vids[0] if vids else None))
    vid_update = gr.update(value=None, visible=bool(vids))
    _log_step("do_scan:end")
    return dd_update, vid_update


# ---------------------
# Helper: toggle provider panels visibility
# ---------------------
def toggle_provider_panels(provider: str):
    _log_step("toggle_provider_panels", provider=provider)
    p = (provider or "").lower()
    return (
        gr.update(visible=(p == "openai")),
        gr.update(visible=(p == "anthropic")),
    )


def on_select(video, outputs_root):
    _log_step("on_select:start", video=video, outputs_root=outputs_root)
    vp_update = gr.update(value=video, visible=True)
    matches = find_existing_outputs_for_video(video, outputs_root)
    _log_step("on_select:found_outputs", count=len(matches))
    dd_update = gr.update(choices=matches, value=(matches[0] if matches else None))
    msg = "Found existing outputs:\n" + ("\n".join(matches[:8]) if matches else "None")
    _log_step("on_select:end")
    return vp_update, dd_update, msg


def on_use_existing(selected_outputs, video_path, state_dict,
                    lang, vclass, fps, device, batch, conf, iou, max_det,
                    vlm_base_url, vlm_api_key, vlm_model, vlm_profile, vlm_maxconc, vlm_stream, vlm_stream_mode,
                    window, anchor, embed_model, embed_device,
                    use_detection
                    ):
    _log_step("on_use_existing:start", selected_outputs=selected_outputs, video_path=video_path)
    if not selected_outputs:
        logger.warning("on_use_existing:no_outputs_selected")
        raise gr.Error("No outputs folder selected.")
    out_dir = Path(selected_outputs).resolve()
    if not out_dir.exists():
        logger.error("on_use_existing:missing_folder", extra={"path": str(out_dir)})
        raise gr.Error(f"Selected outputs folder doesn't exist: {out_dir}")

    srt_path = find_any_srt(str(out_dir))
    have_srt = srt_path is not None
    have_vlm = has_visual_artifacts(str(out_dir))
    existing_idx = find_existing_indexes(str(out_dir))
    idx_dir_trans = existing_idx.get("transcript")
    idx_dir_vlm   = existing_idx.get("visual")
    _log_step("on_use_existing:artifacts_detected",
              have_srt=have_srt, have_vlm=have_vlm,
              idx_trans=bool(idx_dir_trans), idx_vlm=bool(idx_dir_vlm))

    if not idx_dir_trans:
        if not have_srt:
            audio_path = find_audio_file_in(str(out_dir))
            _log_step("on_use_existing:audio_probe", audio_path=str(audio_path) if audio_path else None)
            if audio_path is not None:
                out_dir_str = run_extractor(
                    video_path=str(audio_path),
                    lang=lang, vclass=vclass, fps=(None if not fps else int(fps)),
                    device=device, batch_size=int(batch),
                    conf=float(conf), iou=float(iou), max_det=int(max_det),
                    progress=None,
                    enable_detection=bool(use_detection),
                )
                _log_step("on_use_existing:extractor_done", out_dir_str=out_dir_str)
                if Path(out_dir_str).resolve() != out_dir:
                    out_dir = Path(out_dir_str).resolve()
                srt_path = find_any_srt(str(out_dir))
                have_srt = srt_path is not None
                _log_step("on_use_existing:srt_after_extract", have_srt=have_srt)

        if have_srt and not idx_dir_trans:
            _log_step("on_use_existing:build_transcript_index:start", srt=str(srt_path))
            idx_dir_trans = build_index_from_srt(
                transcript_path=str(srt_path),
                window=int(window), anchor=anchor,
                embed_model=embed_model,
                device=(None if embed_device == "auto" else embed_device),
                progress=None,
            )
            _log_step("on_use_existing:build_transcript_index:done", idx_dir_trans=idx_dir_trans)

    if have_vlm and not idx_dir_vlm:
        _log_step("on_use_existing:build_visual_index:start")
        idx_dir_vlm = build_visual_index(
            outputs_dir=str(out_dir),
            embed_model=embed_model,
            embed_device=embed_device,
        )
        _log_step("on_use_existing:build_visual_index:done", idx_dir_vlm=idx_dir_vlm)

    if not idx_dir_trans or not idx_dir_vlm:
        existing_idx = find_existing_indexes(str(out_dir))
        idx_dir_trans = idx_dir_trans or existing_idx.get("transcript")
        idx_dir_vlm   = idx_dir_vlm   or existing_idx.get("visual")
        _log_step("on_use_existing:refresh_indexes",
                  idx_trans=idx_dir_trans, idx_vlm=idx_dir_vlm)

    if idx_dir_trans: _ = load_index(idx_dir_trans)
    if idx_dir_vlm:   _ = load_index(idx_dir_vlm)

    sd = state_dict or {}
    rec = sd.get(str(video_path), {"out_dir": str(out_dir), "index_dirs": {}})
    rec["out_dir"] = str(out_dir)
    if idx_dir_trans: rec["index_dirs"]["transcript"] = idx_dir_trans
    if idx_dir_vlm:   rec["index_dirs"]["visual"] = idx_dir_vlm
    sd[str(video_path)] = rec
    _log_step("on_use_existing:state_saved", out_dir=str(out_dir),
              has_trans=bool(idx_dir_trans), has_vlm=bool(idx_dir_vlm))

    parts = [f"Using outputs: {out_dir}"]
    parts.append(f"- Transcript (.srt found): {'✔' if have_srt else '✖'}")
    parts.append(f"- Transcript index: {idx_dir_trans or '(none — click Generate if needed)'}")
    parts.append(f"- Visual artifacts (csv/context/meta): {'✔' if have_vlm else '✖'}")
    parts.append(f"- Visual index: {idx_dir_vlm or '(none — click Generate if needed)'}")
    if not have_srt and not idx_dir_trans:
        parts.append("• No transcript SRT or index found. If you didn’t drop an audio file in this folder, click “Generate”.")
    parts.append(f"- transcript index: {idx_dir_trans or '(none)'}")
    parts.append(f"- visual index:     {idx_dir_vlm   or '(none)'}")
    _log_step("on_use_existing:end")
    return "\n".join(parts), sd


def do_generate(
    folder, video_path, outputs_root,
    lang, vclass, fps, device, batch, conf, iou, max_det,
    window, anchor, embed_model, embed_device, state_dict,
    vlm_base_url, vlm_api_key, vlm_model, vlm_profile, vlm_maxconc, vlm_stream, vlm_stream_mode,
    use_transcript: bool, use_detection: bool, use_visual: bool,
    progress=gr.Progress(track_tqdm=True)
):
    _log_step("do_generate:start",
              video_path=video_path, outputs_root=outputs_root,
              use_transcript=use_transcript, use_detection=use_detection, use_visual=use_visual)
    if not video_path:
        logger.warning("do_generate:no_video")
        raise gr.Error("Please select a video.")
    progress(0.0, desc="Starting…")

    existing = find_existing_outputs_for_video(video_path, outputs_root)
    out_dir = existing[0] if existing else None
    if not out_dir:
        stem = safe_stem(Path(video_path))
        out_dir = str(Path(outputs_root).expanduser() / f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    _log_step("do_generate:outputs_dir", out_dir=out_dir)

    progress(0.1, desc="Launching pipelines in parallel…")
    par_res = run_parallel_pipelines(
        want_transcript=bool(use_transcript),
        want_detection=bool(use_detection),
        want_vlm=bool(use_visual),
        video_path=video_path,
        out_dir=out_dir,
        lang=lang, vclass=vclass, fps=fps, device=device, batch=batch, conf=conf, iou=iou, max_det=max_det,
        vlm_base_url=vlm_base_url, vlm_api_key=vlm_api_key, vlm_model=vlm_model,
        vlm_profile=vlm_profile, vlm_maxconc=vlm_maxconc, vlm_stream=vlm_stream, vlm_stream_mode=vlm_stream_mode,
    )
    _log_step("do_generate:parallel_done",
              extractor_ok=par_res["extractor"]["ok"], extractor_err=par_res["extractor"]["error"],
              vlm_ok=par_res["vlm"]["ok"], vlm_err=par_res["vlm"]["error"])

    if par_res["extractor"]["out_dir"]:
        out_dir = par_res["extractor"]["out_dir"]
        _log_step("do_generate:out_dir_switched_by_extractor", out_dir=out_dir)

    idx_dir_trans = None
    if use_transcript:
        if par_res["extractor"]["error"]:
            logger.error("do_generate:extractor_error %s", par_res["extractor"]["error"])
            progress(0.35, desc="Extractor error (transcript)")
        else:
            srt_path = par_res["extractor"]["srt_path"]
            _log_step("do_generate:srt_probe", has_srt=bool(srt_path), srt_path=srt_path)
            if not srt_path:
                progress(0.35, desc="Transcript requested but no .srt produced.")
            else:
                progress(0.55, desc="Building transcript index…")
                _log_step("do_generate:build_transcript_index:start")
                idx_dir_trans = build_index_from_srt(
                    transcript_path=str(srt_path),
                    window=int(window), anchor=anchor,
                    embed_model=embed_model,
                    device=(None if embed_device == "auto" else embed_device),
                    progress=None,
                )
                _log_step("do_generate:build_transcript_index:done", idx_dir_trans=idx_dir_trans)
                progress(0.7, desc="Transcript index ready.")

    idx_dir_vlm = None
    if use_visual:
        if par_res["vlm"]["error"]:
            logger.error("do_generate:vlm_error %s", par_res["vlm"]["error"])
            progress(0.75, desc=f"VLM error: {par_res['vlm']['error']}")
        else:
            have_vlm = has_visual_artifacts(out_dir)
            _log_step("do_generate:vlm_artifacts_probe", have_vlm=have_vlm)
            if not have_vlm:
                progress(0.8, desc="VLM artifacts not found after extraction.")
            else:
                progress(0.86, desc="Building visual index…")
                _log_step("do_generate:build_visual_index:start")
                idx_dir_vlm = build_visual_index(
                    outputs_dir=out_dir,
                    embed_model=embed_model,
                    embed_device=embed_device,
                )
                _log_step("do_generate:build_visual_index:done", idx_dir_vlm=idx_dir_vlm)
                progress(0.92, desc="Visual index ready.")

    sd = state_dict or {}
    rec = {"out_dir": out_dir, "index_dirs": {}}
    if idx_dir_trans: rec["index_dirs"]["transcript"] = idx_dir_trans
    if idx_dir_vlm:   rec["index_dirs"]["visual"] = idx_dir_vlm
    sd[str(video_path)] = rec
    _log_step("do_generate:state_saved", out_dir=out_dir,
              has_trans=bool(idx_dir_trans), has_vlm=bool(idx_dir_vlm))

    progress(1.0, desc="Done.")

    status = [f"✅ Prepared in: {out_dir}"]
    status.append(f"- Transcript index: {idx_dir_trans or '(skipped or failed)'}")
    status.append(f"- Visual index:     {idx_dir_vlm or '(skipped or failed)'}")
    status.append(f"- Object detection: {'enabled' if use_detection else 'disabled'}")

    if par_res["extractor"]["error"]:
        status.append(f"⚠️ Extractor error: {par_res['extractor']['error']}")
    if par_res["vlm"]["error"]:
        status.append(f"⚠️ VLM error: {par_res['vlm']['error']}")

    _log_step("do_generate:end")
    return "\n".join(status), sd


def _provider_cfg(provider,
                  oa_base_url, oa_api_key, oa_model,
                  an_api_key, an_model) -> Dict:
    p = (provider or "anthropic").lower()
    if p == "openai":
        return {
            "oa_base_url": (oa_base_url or "").strip(),
            "oa_api_key": (oa_api_key or "").strip(),
            "oa_model": (oa_model or "").strip(),
        }
    elif p == "anthropic":
        return {
            "an_api_key": (an_api_key or "").strip(),
            "an_model": (an_model or "").strip(),
        }
    else:
        return {}


def _validate_provider_inputs(provider: str, cfg: Dict) -> Optional[str]:
    p = (provider or "anthropic").lower()
    if p == "openai":
        if not cfg.get("oa_api_key") or not cfg.get("oa_model"):
            return "Please set OpenAI API Key and Model in the panel (Base URL optional)."
    elif p == "anthropic":
        if not cfg.get("an_api_key") or not cfg.get("an_model"):
            return "Please set Anthropic API Key and Model in the panel."
    return None


def _yield_stream_with_sanitize(
    stream_gen,
    msgs,
    radio_update,
    label_to_start,
    ctx_html_string,
    allowed_nums,
    times_by_num,
):
    """
    Consume a token stream, yield partial chat updates,
    then do a final sanitize pass and yield once more.
    """
    acc = ""
    msgs.append({"role": "assistant", "content": ""})
    for chunk in stream_gen:
        if not chunk:
            continue
        acc += chunk
        msgs[-1]["content"] = acc
        # live partial update
        yield msgs, radio_update, label_to_start, ctx_html_string

    # final sanitize + final update
    msgs[-1]["content"] = sanitize_scene_output(acc, allowed_nums, times_by_num)
    yield msgs, radio_update, label_to_start, ctx_html_string


def intercept_or_stream(
    *,
    dispatch_mode: str,
    provider: str,
    cfg: Dict,
    messages: List[Dict],
    temperature: float,
    max_tokens: int,
    msgs_visible_chat: List[Dict],     # current visible chat array (mutable)
    radio_update, label_to_start, ctx_html_string,
    allowed_nums: set, times_by_num: dict,
):
    """
    If dispatch_mode == 'Check before', prepare a preview and return a single yield
    that sets pending payload + shows preview. Else, start streaming and yield progressively.
    """
    if (dispatch_mode or "").lower().startswith("check"):
        preview_text, total_chars, approx_tokens = messages_preview_text(messages)
        safe_cfg = redact_cfg_for_preview(cfg)
        preview_html = (
            "<h3>LLM payload preview</h3>"
            f"<p><b>Total characters:</b> {total_chars} &nbsp; "
            f"<b>Approx. tokens:</b> {approx_tokens}</p>"
            "<pre style='white-space:pre-wrap;'>"
            + html_escape(preview_text)
            + "</pre>"
            "<p><em>Provider config (redacted):</em></p>"
            "<pre style='white-space:pre-wrap;'>"
            + html_escape(json.dumps(safe_cfg, indent=2, ensure_ascii=False))
            + "</pre>"
        )

        # Pack everything needed to resume streaming later on "Send to LLM now"
        pending = {
            "provider": provider,
            "cfg": cfg,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # carry UI artifacts so the approval flow can render identically
            "radio_update": radio_update,
            "label_to_start": label_to_start,
            "ctx_html_string": ctx_html_string,
            "allowed_nums": list(allowed_nums),
            "times_by_num": times_by_num,
        }

        msgs_visible_chat.append({
            "role": "assistant",
            "content": "Preview ready. Review the payload below, then click “Send to LLM now”."
        })
        # single final yield for the interception
        yield (
            msgs_visible_chat,
            radio_update,
            label_to_start,
            ctx_html_string,
            gr.update(value=preview_html, visible=True),
            pending,
            gr.update(visible=True),   # show 'Send to LLM now'
        )
        return

    # Auto send: start streaming immediately
    stream = llm_chat_stream(
        provider=(provider or "anthropic"),
        cfg=cfg,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    acc = ""
    msgs_visible_chat.append({"role": "assistant", "content": ""})
    for chunk in stream:
        if not chunk:
            continue
        acc += chunk
        msgs_visible_chat[-1]["content"] = acc
        yield (
            msgs_visible_chat,
            radio_update,
            label_to_start,
            ctx_html_string,
            gr.update(value="", visible=False),
            None,
            gr.update(visible=False),
        )
    msgs_visible_chat[-1]["content"] = sanitize_scene_output(acc, allowed_nums, times_by_num)
    yield (
        msgs_visible_chat,
        radio_update,
        label_to_start,
        ctx_html_string,
        gr.update(value="", visible=False),
        None,
        gr.update(visible=False),
    )


def on_chat(
    user_msg, history,
    video_path, ctx_before, ctx_after, top_k, method, alpha,
    rerank, rerank_model, overfetch,
    provider,
    oa_base_url, oa_api_key, oa_model,
    an_api_key, an_model,
    embed_device, embed_model_override,
    enable_web, exa_api_key, exa_num_results,
    state_dict,
    source_mode,
    dispatch_mode,
):
    """
    Generator that yields:
      (messages, ts_radio_update, ts_map_state, ctx_html_string, preview_html_update, pending_obj, send_now_btn_update)
    """

    _log_step("on_chat:start",
              video_path=video_path, provider=provider, source_mode=source_mode,
              dispatch_mode=dispatch_mode, top_k=top_k, method=method, rerank=rerank)

    # visible chat
    msgs = list(history or [])

    # quick helpers for consistent trailing outputs
    def _no_preview():
        return gr.update(value="", visible=False), None, gr.update(visible=False)

    def _show_preview(preview_html, pending_obj):
        return gr.update(value=preview_html, visible=True), pending_obj, gr.update(visible=True)

    # basic validations
    if not video_path:
        logger.warning("on_chat:no_video")
        msgs.append({"role": "assistant", "content": "Please select a video first."})
        yield msgs, gr.update(choices=[], value=None, visible=False), {}, "", *_no_preview()
        return

    cfg = _provider_cfg(provider, oa_base_url, oa_api_key, oa_model, an_api_key, an_model)
    err = _validate_provider_inputs(provider, cfg)
    if err:
        logger.warning("on_chat:provider_validation_failed %s", err)
        msgs.append({"role": "assistant", "content": err})
        yield msgs, gr.update(choices=[], value=None, visible=False), {}, "", *_no_preview()
        return

    rec = (state_dict or {}).get(str(video_path))
    if not rec:
        logger.warning("on_chat:no_outputs_mapped")
        msgs.append({"role": "assistant", "content": "No outputs mapped for this video. Click 'Use selected outputs' or 'Generate'."})
        yield msgs, gr.update(choices=[], value=None, visible=False), {}, "", *_no_preview()
        return

    # Append latest user msg
    latest_user = str(user_msg)
    if not (msgs and msgs[-1].get("role") == "user" and msgs[-1].get("content") == latest_user):
        msgs.append({"role": "user", "content": latest_user})

    _log_step("on_chat:user_message", length=len(latest_user))

    # History windows
    history_window_for_rewrite = trim_history_messages(msgs[:-1] or [], max_turns=10, max_chars=6000)
    history_text_for_router   = format_history_as_text(msgs[:-1] or [], max_turns=10, max_chars=6000)

    # (1) contextualize
    try:
        standalone_q = contextualize_question(
            provider=(provider or "anthropic"),
            cfg=cfg,
            history_msgs=history_window_for_rewrite,
            latest_user_msg=latest_user,
        )
        _log_step("on_chat:contextualized_question", length=len(standalone_q))
    except Exception as exp:
        logger.exception(f"on_chat:contextualize_question_error {exp}")
        standalone_q = latest_user

    # ---------- ENTITY (simple path) ----------
    is_entity, entity_name = detect_entity_intent(standalone_q)
    _log_step("on_chat:entity_detection", is_entity=is_entity, entity_name=entity_name)
    if is_entity and entity_name:
        out_dir = (state_dict or {}).get(str(video_path), {}).get("out_dir")
        entities_scenes, norm_map = load_entities_scenes_from_context(out_dir) if out_dir else ({}, {})
        resolved_key = fuzzy_pick_entity(entity_name, norm_map)
        _log_step("on_chat:entity_resolved", resolved_key=resolved_key, has_context=bool(entities_scenes))

        if entities_scenes and resolved_key in entities_scenes:
            scene_ids = sorted({int(s) for s in entities_scenes.get(resolved_key, [])})
            _log_step("on_chat:entity_scenes_found", count=len(scene_ids))
            scene_rows = load_scene_rows_from_csv(out_dir, scene_ids)
            hits = hits_from_scene_rows(scene_rows)
            ctx_text = "\n\n".join(
                [f"[Scene {r['scene_id']}] {r['start_timecode']}–{r['end_timecode']}\n{r['generated_text']}" for r in scene_rows]
            )
            ctx_html_string = ctx_md_from_hits_aggregated(hits, title=f"Scenes for entity: {resolved_key}")

            # timecodes radio
            labels, label_to_start = [], {}
            for h in hits:
                c0 = h["context"][0]
                snippet = (c0.get("text","") or "").replace("\n"," ")
                if len(snippet) > 100: snippet = snippet[:97] + "..."
                label = f"[VLM] {c0['start_srt']} - {c0['end_srt']} | {snippet}"
                labels.append(label)
                label_to_start[label] = srt_to_seconds(c0["start_srt"])
            radio_update = gr.update(choices=labels, value=None, visible=bool(labels))

            # allow-list for sanitize
            allowed = []
            for r in scene_rows:
                allowed.append({"num": int(r["scene_id"]), "start": r["start_timecode"], "end": r["end_timecode"], "src": "VLM"})
            allowed_nums = {a["num"] for a in allowed}
            times_by_num = {a["num"]: (a["start"], a["end"]) for a in allowed}

            system = (
                "Tu es un assistant vidéo. Règles STRICTES:\n"
                "• Pour chaque élément, préfixe le timecode EXACT avec **SCENE N°{num}: {HH:MM:SS,mmm–HH:MM:SS,mmm}**.\n"
                "• {num} = scene_id s'il est fourni, sinon le numéro du passage (rank).\n"
                "• 1–2 lignes par scène. N'invente rien au-delà du contexte fourni."
                f"ALLOWED_SCENES = {json.dumps(allowed, ensure_ascii=False)}"
            )
            user_full = (
                f"Requête utilisateur: Donne toutes les scènes où «{entity_name}» apparaît.\n"
                f"Entité résolue (normalisée): {resolved_key}\n"
                f"Scenes trouvées: {scene_ids}\n\n"
                f"Contexte par scène (cartes VLM):\n{ctx_text}\n\n"
                "Tâche: Résume et liste les scènes sous forme de puces: [scene_id] HH:MM:SS,mmm–HH:MM:SS,mmm — 1 à 2 lignes utiles."
            )
            messages_payload = [{"role":"system","content":system},{"role":"user","content":user_full}]

            # Dispatch mode
            if (dispatch_mode or "").lower().startswith("check"):
                preview_text, total_chars, approx_tokens = messages_preview_text(messages_payload)
                _log_step("on_chat:preview_ready", branch="entity_simple", total_chars=total_chars, approx_tokens=approx_tokens)
                safe_cfg = redact_cfg_for_preview(cfg)
                preview_html = (
                    "<h3>LLM payload preview</h3>"
                    f"<p><b>Total characters:</b> {total_chars} &nbsp; <b>Approx. tokens:</b> {approx_tokens}</p>"
                    "<pre style='white-space:pre-wrap;'>"+html_escape(preview_text)+"</pre>"
                    "<p><em>Provider config (redacted):</em></p>"
                    "<pre style='white-space:pre-wrap;'>"+html_escape(json.dumps(safe_cfg, indent=2, ensure_ascii=False))+"</pre>"
                )
                pending = {
                    "provider": provider,
                    "cfg": cfg,
                    "messages": messages_payload,
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "radio_update": radio_update,
                    "label_to_start": label_to_start,
                    "ctx_html_string": ctx_html_string,
                    "allowed_nums": list(allowed_nums),
                    "times_by_num": times_by_num,
                }
                msgs.append({"role": "assistant", "content": "Preview ready. Review below, then click “Send to LLM now”."})
                yield msgs, radio_update, label_to_start, ctx_html_string, *_show_preview(preview_html, pending)
                _log_step("on_chat:end", branch="entity_simple_preview")
                return

            # Auto send: stream now
            try:
                _log_step("on_chat:stream_start", branch="entity_simple", temperature=0.1, max_tokens=2000)
                stream = llm_chat_stream(
                    provider=provider, cfg=cfg,
                    messages=messages_payload,
                    temperature=0.1, max_tokens=2000
                )
                acc = ""
                msgs.append({"role": "assistant", "content": ""})
                for chunk in stream:
                    if not chunk: continue
                    acc += chunk
                    msgs[-1]["content"] = acc
                    yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
                msgs[-1]["content"] = sanitize_scene_output(acc, allowed_nums, times_by_num)
                _log_step("on_chat:stream_end", branch="entity_simple", total_chars=len(acc))
                yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
                _log_step("on_chat:end", branch="entity_simple_streamed")
                return
            except Exception as exp:
                logger.exception(f"on_chat:entity_simple_stream_error {exp}")
                msgs.append({"role": "assistant", "content": "LLM error during streaming."})
                yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
                return

    # ---------- ENTITY (router + key matching) ----------
    try:
        intent_res = llm_detect_intent_entities(
            provider=(provider or "anthropic"),
            cfg=cfg,
            latest_user_msg=str(standalone_q),
            history_msgs=(history or []),
        )
        _log_step("on_chat:intent_entities", intent=intent_res.get("intent"), entities=intent_res.get("entities", []))
    except Exception as exp:
        logger.exception(f"on_chat:intent_router_error {exp}")
        intent_res = {"intent":"other","entities":[],"reason":"router failed"}

    if intent_res.get("intent") == "entity_search" and intent_res.get("entities"):
        out_dir = (state_dict or {}).get(str(video_path), {}).get("out_dir")
        entities_scenes, candidate_keys = load_entities_scenes_from_context(out_dir) if out_dir else ({}, [])
        _log_step("on_chat:entity_router_candidates", candidates=len(candidate_keys))
        if entities_scenes and candidate_keys:
            try:
                match_res = llm_match_entities_to_keys(
                    provider=(provider or "anthropic"),
                    cfg=cfg,
                    query_entities=intent_res["entities"],
                    candidate_keys=candidate_keys,
                )
                _log_step("on_chat:entity_router_matches", matches=len(match_res.get("matches", [])), flat_keys=len((match_res.get("flat_keys") or [])))
            except Exception as exp:
                logger.exception(f"on_chat:entity_key_match_error {exp}")
                match_res = {"matches": [], "flat_keys": []}

            matched_keys: List[str] = match_res.get("flat_keys") or []
            if matched_keys:
                scene_ids = sorted({int(s) for k in matched_keys for s in (entities_scenes.get(k) or [])})
                _log_step("on_chat:entity_router_scene_ids", count=len(scene_ids))
                scene_rows = load_scene_rows_from_csv(out_dir, scene_ids)
                hits = hits_from_scene_rows(scene_rows)
                ctx_text = "\n\n".join(
                    [f"[Scene {r['scene_id']}] {r['start_timecode']}–{r['end_timecode']}\n{r['generated_text']}" for r in scene_rows]
                )
                ctx_html_string = ctx_md_from_hits_aggregated(hits, title=f"Scenes for entities: {', '.join(matched_keys)}")

                labels, label_to_start = [], {}
                for h in hits:
                    c0 = h["context"][0]
                    snippet = (c0.get("text","") or "").replace("\n"," ")
                    if len(snippet) > 100: snippet = snippet[:97] + "..."
                    label = f"[VLM] {c0['start_srt']} - {c0['end_srt']} | {snippet}"
                    labels.append(label)
                    label_to_start[label] = srt_to_seconds(c0["start_srt"])
                radio_update = gr.update(choices=labels, value=None, visible=bool(labels))

                allowed = [{"num": int(r["scene_id"]), "start": r["start_timecode"], "end": r["end_timecode"], "src": "VLM"} for r in scene_rows]
                allowed_nums = {a["num"] for a in allowed}
                times_by_num = {a["num"]: (a["start"], a["end"]) for a in allowed}

                system = (
                    "Tu es un assistant vidéo. Règles STRICTES:\n"
                    "• Pour chaque scène, affiche **SCENE N°{num}: {HH:MM:SS,mmm–HH:MM:SS,mmm}** suivi d'une brève explication.\n"
                    "• {num} = scene_id si présent, sinon le numéro du passage (rank).\n"
                    "• N'invente rien."
                    f"ALLOWED_SCENES = {json.dumps(allowed, ensure_ascii=False)}"
                )
                user_full = json.dumps({
                    "user_query": str(standalone_q),
                    "query_entities": intent_res["entities"],
                    "matched_keys": matched_keys,
                    "scenes": scene_ids,
                    "vlm_cards_text": ctx_text,
                }, ensure_ascii=False)
                messages_payload = [{"role":"system","content":system},{"role":"user","content":user_full}]

                if (dispatch_mode or "").lower().startswith("check"):
                    preview_text, total_chars, approx_tokens = messages_preview_text(messages_payload)
                    _log_step("on_chat:preview_ready", branch="entity_router", total_chars=total_chars, approx_tokens=approx_tokens)
                    safe_cfg = redact_cfg_for_preview(cfg)
                    preview_html = (
                        "<h3>LLM payload preview</h3>"
                        f"<p><b>Total characters:</b> {total_chars} &nbsp; <b>Approx. tokens:</b> {approx_tokens}</p>"
                        "<pre style='white-space:pre-wrap;'>"+html_escape(preview_text)+"</pre>"
                        "<p><em>Provider config (redacted):</em></p>"
                        "<pre style='white-space:pre-wrap;'>"+html_escape(json.dumps(safe_cfg, indent=2, ensure_ascii=False))+"</pre>"
                    )
                    pending = {
                        "provider": provider,
                        "cfg": cfg,
                        "messages": messages_payload,
                        "temperature": 0.1,
                        "max_tokens": 700,
                        "radio_update": radio_update,
                        "label_to_start": label_to_start,
                        "ctx_html_string": ctx_html_string,
                        "allowed_nums": list(allowed_nums),
                        "times_by_num": times_by_num,
                    }
                    msgs.append({"role": "assistant", "content": "Preview ready. Review below, then click “Send to LLM now”."})
                    yield msgs, radio_update, label_to_start, ctx_html_string, *_show_preview(preview_html, pending)
                    _log_step("on_chat:end", branch="entity_router_preview")
                    return

                try:
                    _log_step("on_chat:stream_start", branch="entity_router", temperature=0.1, max_tokens=700)
                    stream = llm_chat_stream(
                        provider=(provider or "anthropic"),
                        cfg=cfg,
                        messages=messages_payload,
                        temperature=0.1,
                        max_tokens=700,
                    )
                    acc = ""
                    msgs.append({"role": "assistant", "content": ""})
                    for chunk in stream:
                        if not chunk: continue
                        acc += chunk
                        msgs[-1]["content"] = acc
                        yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
                    msgs[-1]["content"] = sanitize_scene_output(acc, allowed_nums, times_by_num)
                    _log_step("on_chat:stream_end", branch="entity_router", total_chars=len(acc))
                    yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
                    _log_step("on_chat:end", branch="entity_router_streamed")
                    return
                except Exception as exp:
                    logger.exception(f"on_chat:entity_router_stream_error {exp}")
                    msgs.append({"role":"assistant","content": f"LLM error: {exp}"})
                    yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
                    return

    # ---------- Retrieval selection ----------
    available = (state_dict or {}).get(str(video_path), {}).get("index_dirs", {})
    have_trans = "transcript" in available
    have_visual = "visual" in available

    mode = (source_mode or "auto").lower()
    if mode == "both":
        want_transcript, want_visual = True, True
    elif mode == "transcript":
        want_transcript, want_visual = True, False
    elif mode == "visual":
        want_transcript, want_visual = False, True
    else:
        want_transcript, want_visual, _reason = auto_select_sources_from_query(standalone_q)

    if want_transcript and not have_trans: want_transcript = False
    if want_visual and not have_visual:   want_visual = False
    if not want_transcript and not want_visual:
        if have_trans:   want_transcript = True
        elif have_visual: want_visual = True
        else:
            _log_step("on_chat:no_indexes_available")
            msgs.append({"role": "assistant", "content": "No RAG indexes are available for this video. Generate or load indexes first."})
            yield msgs, gr.update(choices=[], value=None, visible=False), {}, "", *_no_preview()
            return

    idx_pairs = []
    if want_transcript and have_trans: idx_pairs.append((load_index(available["transcript"]), "SRT"))
    if want_visual and have_visual:     idx_pairs.append((load_index(available["visual"]), "VLM"))

    hits, ctx_text = compile_context_blocks_multi(
        indexes=idx_pairs,
        query=str(standalone_q), top_k=int(top_k), method=method, alpha=float(alpha),
        rerank=rerank, rerank_model=rerank_model, overfetch=int(overfetch),
        ctx_before=int(ctx_before), ctx_after=int(ctx_after),
        device=(None if embed_device == "auto" else embed_device),
        embed_model_override=(None if not embed_model_override else embed_model_override)
    )

    allowed = []
    for h in hits:
        num = int(h.get("scene_id", h.get("rank", 0)) or h.get("rank", 0))
        ctx_sorted = sorted(h["context"], key=lambda c: (c.get("offset",0)!=0, c.get("offset",0)))
        main = next((c for c in ctx_sorted if c.get("offset",0)==0), ctx_sorted[0])
        allowed.append({
            "num": num,
            "start": main.get("start_srt","00:00:00,000"),
            "end":   main.get("end_srt","00:00:00,000"),
            "src":   h.get("source","srt").upper(),
        })
    allowed_nums = {a["num"] for a in allowed}
    times_by_num = {a["num"]: (a["start"], a["end"]) for a in allowed}

    ctx_html_string = ctx_md_from_hits_aggregated(hits, title="Retrieved passages")

    labels, label_to_start = [], {}
    for h in hits:
        ctx_sorted = sorted(h["context"], key=lambda c: (c["offset"] != 0, c["offset"]))
        main = next((c for c in ctx_sorted if c.get("offset", 0) == 0), ctx_sorted[0])
        start_srt = main.get("start_srt", h.get("start_srt", "00:00:00,000"))
        end_srt   = main.get("end_srt",   h.get("end_srt",   "00:00:00,000"))
        snippet = (main.get("text", "") or "").replace("\n", " ")
        if len(snippet) > 100: snippet = snippet[:97] + "..."
        source_label = format_source_label(h.get("source"))
        label = f"[{source_label}] {start_srt} - {end_srt} | {snippet}".strip()
        labels.append(label)
        label_to_start[label] = srt_to_seconds(start_srt)
    radio_update = gr.update(choices=labels, value=None, visible=bool(labels))

    _log_step("on_chat:retrieval_choice",
              want_transcript=want_transcript, want_visual=want_visual,
              have_trans=have_trans, have_visual=have_visual)

    _log_step("on_chat:retrieved_passages",
              hits=len(hits), ctx_chars=len(ctx_text))

    # (3) Router
    explicit_flag = wants_web_search_explicit(latest_user)
    cy = datetime.now().year
    rec_flag1, yrs1 = wants_recent_info(latest_user, cy)
    rec_flag2, yrs2 = wants_recent_info(standalone_q, cy)
    recency_flag = rec_flag1 or rec_flag2
    years_mentioned = sorted(set(yrs1 + yrs2))

    _log_step("on_chat:recency", explicit_flag=explicit_flag,
              recency_flag=recency_flag, years=years_mentioned)

    if recency_flag and (not enable_web or not exa_api_key):
        logger.info("on_chat:recency_block_no_web")
        msgs.append({"role":"assistant",
            "content":"This looks time-sensitive (e.g., 'now/2025'). Enable web search (Exa) to fetch up-to-date info, otherwise I can only answer from the video’s content."})
        msgs[-1]["content"] = sanitize_scene_output(msgs[-1]["content"], allowed_nums, times_by_num)
        yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
        _log_step("on_chat:end", branch="recency_block")
        return

    try:
        decision = llm_decide_search(
            provider=(provider or "anthropic"),
            cfg=cfg,
            question=str(standalone_q),
            transcript_context=ctx_text,
            explicit_flag=explicit_flag,
            history_text=history_text_for_router,
            recency_flag=recency_flag,
            years_mentioned=years_mentioned,
            current_year=cy,
        )
        _log_step("on_chat:router_decision",
                  need_search=bool(decision.get("need_search")),
                  query=decision.get("query"),
                  answer_present=bool(decision.get("answer")))
    except Exception as e:
        logger.exception(f"on_chat:router_error {e}")
        msgs.append({"role": "assistant", "content": f"Routing error: {e}"})
        msgs[-1]["content"] = sanitize_scene_output(msgs[-1]["content"], allowed_nums, times_by_num)
        yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
        return

    need_search = bool(decision.get("need_search"))
    search_query = decision.get("query")
    direct_answer = decision.get("answer")  # not used directly; we still call model

    # (4A) No web search → answer from transcript + history
    if not need_search:
        history_for_model = trim_history_messages(msgs[:-1], max_turns=10, max_chars=6000)
        system = (
            "Tu es un assistant qui répond en combinant:\n"
            "1) le transcript (fiable pour les propos et timestamps),\n"
            "2) l'historique de la conversation.\n"
            "RÈGLE DE FORMATAGE OBLIGATOIRE:\n"
            "• Chaque timecode cité doit être préfixé par **SCENE N°{num}: {HH:MM:SS,mmm–HH:MM:SS,mmm}**.\n"
            "• {num} = scene_id si présent, sinon le numéro du passage (rank).\n"
            "N'invente rien au-delà du transcript."
            f"ALLOWED_SCENES = {json.dumps(allowed, ensure_ascii=False)}"
        )
        user_full = (
            f"Dernière question (standalone):\n{standalone_q}\n\n"
            f"Transcript context (passages):\n{ctx_text}\n"
        )
        messages_payload = [{"role": "system", "content": system}, *history_for_model, {"role": "user", "content": user_full}]

        if (dispatch_mode or "").lower().startswith("check"):
            preview_text, total_chars, approx_tokens = messages_preview_text(messages_payload)
            _log_step("on_chat:preview_ready", branch="no_web", total_chars=total_chars, approx_tokens=approx_tokens)
            safe_cfg = redact_cfg_for_preview(cfg)
            preview_html = (
                "<h3>LLM payload preview</h3>"
                f"<p><b>Total characters:</b> {total_chars} &nbsp; <b>Approx. tokens:</b> {approx_tokens}</p>"
                "<pre style='white-space:pre-wrap;'>"+html_escape(preview_text)+"</pre>"
                "<p><em>Provider config (redacted):</em></p>"
                "<pre style='white-space:pre-wrap;'>"+html_escape(json.dumps(safe_cfg, indent=2, ensure_ascii=False))+"</pre>"
            )
            pending = {
                "provider": provider,
                "cfg": cfg,
                "messages": messages_payload,
                "temperature": 0.2,
                "max_tokens": 900,
                "radio_update": radio_update,
                "label_to_start": label_to_start,
                "ctx_html_string": ctx_html_string,
                "allowed_nums": list(allowed_nums),
                "times_by_num": times_by_num,
            }
            msgs.append({"role": "assistant", "content": "Preview ready. Review below, then click “Send to LLM now”."})
            yield msgs, radio_update, label_to_start, ctx_html_string, *_show_preview(preview_html, pending)
            _log_step("on_chat:end", branch="no_web_preview")
            return

        try:
            _log_step("on_chat:stream_start", branch="no_web", temperature=0.2, max_tokens=900)
            stream = llm_chat_stream(
                provider=(provider or "anthropic"),
                cfg=cfg,
                messages=messages_payload,
                temperature=0.2,
                max_tokens=900,
            )
            acc = ""
            msgs.append({"role": "assistant", "content": ""})
            for chunk in stream:
                if not chunk: continue
                acc += chunk
                msgs[-1]["content"] = acc
                yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
            msgs[-1]["content"] = sanitize_scene_output(acc, allowed_nums, times_by_num)
            _log_step("on_chat:stream_end", branch="no_web", total_chars=len(acc))
            yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
            _log_step("on_chat:end", branch="no_web_streamed")
            return
        except Exception as e:
            logger.exception(f"on_chat:no_web_stream_error {e}")
            msgs.append({"role": "assistant", "content": f"LLM error: {e}"})
            yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
            return

    # (4B) Web search suggested
    if not search_query:
        _log_step("on_chat:web_missing_query")
        msgs.append({"role": "assistant", "content": "La recherche web a été suggérée, mais aucune requête n'a été fournie."})
        yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
        return
    if not enable_web:
        _log_step("on_chat:web_disabled", suggested_query=search_query)
        msgs.append({"role": "assistant", "content": f"🔎 Requête web suggérée : \"{search_query}\" (la recherche web est désactivée)."})
        yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
        return
    if not exa_api_key:
        _log_step("on_chat:web_no_api_key", suggested_query=search_query)
        msgs.append({"role": "assistant", "content": f"🔎 Requête web suggérée : \"{search_query}\" (ajoutez une clé Exa pour effectuer la recherche)."})
        yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
        return

    # status message while searching
    msgs.append({"role": "assistant", "content": f"🔎 Web search query: \"{search_query}\" (running Exa…)"})
    _log_step("on_chat:exa_query", q=search_query, num_results=exa_num_results)
    yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()

    # Exa search
    try:
        web_hits = exa_search_with_contents(search_query, exa_api_key, num_results=int(exa_num_results))
        _log_step("on_chat:exa_results", count=len(web_hits or []))
    except Exception as e:
        logger.exception(f"on_chat:exa_search_failed {e}")
        msgs.append({"role": "assistant", "content": f"Exa search failed: {e}"})
        yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
        return
    if not web_hits:
        _log_step("on_chat:exa_no_results")
        msgs.append({"role": "assistant", "content": "Aucun résultat web pertinent n'a été trouvé."})
        yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
        return

    web_blocks = []
    for i, w in enumerate(web_hits, start=1):
        block = f"[{i}] {w['title']}  {w['url']}\n{w['snippet']}"
        web_blocks.append(block)
    web_context = "\n\n".join(web_blocks)

    system = (
        "Tu es un assistant qui répond en combinant:\n"
        "1) l'historique de la conversation,\n"
        "2) le transcript (fiable pour propos/timestamps),\n"
        "3) des extraits web.\n"
        "RÈGLE DE FORMATAGE OBLIGATOIRE:\n"
        "• Chaque timecode cité doit être préfixé **SCENE N°{num}: {HH:MM:SS,mmm–HH:MM:SS,mmm}** (scene_id si dispo, sinon passage rank).\n"
        "Pour le web, cite les sources en [1], [2], etc."
        f"ALLOWED_SCENES = {json.dumps(allowed, ensure_ascii=False)}"
    )
    user_full = (
        f"Dernière question (standalone):\n{standalone_q}\n\n"
        f"Transcript context:\n{ctx_text}\n\n"
        f"Web results:\n{web_context}"
    )
    history_for_model = trim_history_messages(msgs[:-1], max_turns=10, max_chars=6000)
    messages_payload = [{"role": "system", "content": system}, *history_for_model, {"role": "user", "content": user_full}]

    if (dispatch_mode or "").lower().startswith("check"):
        preview_text, total_chars, approx_tokens = messages_preview_text(messages_payload)
        _log_step("on_chat:preview_ready", branch="web", total_chars=total_chars, approx_tokens=approx_tokens)
        safe_cfg = redact_cfg_for_preview(cfg)
        preview_html = (
            "<h3>LLM payload preview</h3>"
            f"<p><b>Total characters:</b> {total_chars} &nbsp; <b>Approx. tokens:</b> {approx_tokens}</p>"
            "<pre style='white-space:pre-wrap;'>"+html_escape(preview_text)+"</pre>"
            "<p><em>Provider config (redacted):</em></p>"
            "<pre style='white-space:pre-wrap;'>"+html_escape(json.dumps(safe_cfg, indent=2, ensure_ascii=False))+"</pre>"
        )
        pending = {
            "provider": provider,
            "cfg": cfg,
            "messages": messages_payload,
            "temperature": 0.2,
            "max_tokens": 900,
            "radio_update": radio_update,
            "label_to_start": label_to_start,
            "ctx_html_string": ctx_html_string,
            "allowed_nums": list(allowed_nums),
            "times_by_num": times_by_num,
        }
        msgs.append({"role": "assistant", "content": "Preview ready. Review below, then click “Send to LLM now”."})
        yield msgs, radio_update, label_to_start, ctx_html_string, *_show_preview(preview_html, pending)
        _log_step("on_chat:end", branch="web_preview")
        return

    try:
        _log_step("on_chat:stream_start", branch="web", temperature=0.2, max_tokens=900)
        stream = llm_chat_stream(
            provider=(provider or "anthropic"),
            cfg=cfg,
            messages=messages_payload,
            temperature=0.2,
            max_tokens=900,
        )
        acc = ""
        msgs.append({"role": "assistant", "content": ""})
        for chunk in stream:
            if not chunk: continue
            acc += chunk
            msgs[-1]["content"] = acc
            yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
        msgs[-1]["content"] = sanitize_scene_output(acc, allowed_nums, times_by_num)
        _log_step("on_chat:stream_end", branch="web", total_chars=len(acc))
        yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
        _log_step("on_chat:end", branch="web_streamed")
        return
    except Exception as e:
        logger.exception(f"on_chat:web_stream_error {e}")
        msgs.append({"role": "assistant", "content": f"LLM error (final): {e}"})
        yield msgs, radio_update, label_to_start, ctx_html_string, *_no_preview()
        return


def on_upload_video(file_path: str, videos_dir: str):
    _log_step("on_upload_video:start", file_path=file_path, videos_dir=videos_dir)
    if not file_path:
        logger.warning("on_upload_video:no_file")
        raise gr.Error("No file uploaded.")

    src = Path(file_path)
    if not src.exists():
        logger.error("on_upload_video:tmp_missing", path=str(src))
        raise gr.Error("Upload failed: temporary file not found.")

    ext = src.suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        logger.warning("on_upload_video:bad_ext", ext=ext)
        raise gr.Error(f"Invalid file type: {ext}. Allowed: {sorted(ALLOWED_VIDEO_EXTS)}")

    dst_dir = Path(videos_dir).expanduser()
    dst_dir.mkdir(parents=True, exist_ok=True)

    dest = dst_dir / src.name
    if dest.exists():
        dest = dst_dir / f"{src.stem}_{int(time.time())}{src.suffix}"
    shutil.copy2(src, dest)
    _log_step("on_upload_video:copied", dest=str(dest))

    vids = list_videos(str(dst_dir))
    _log_step("on_upload_video:end", folder_count=len(vids))
    return (
        gr.update(choices=vids, value=str(dest)),
        gr.update(value=str(dest), visible=True),
        f"✅ Uploaded to: {dest}"
    )


def on_send_now(
    chat_history,
    pending_payload,
):
    _log_step("on_send_now:start", has_pending=bool(pending_payload))
    msgs = list(chat_history or [])

    if not pending_payload:
        _log_step("on_send_now:nothing_to_send")
        yield (
            msgs,
            gr.update(choices=[], value=None, visible=False),
            {},
            "",
            gr.update(value="", visible=False),
            None,
            gr.update(visible=False),
        )
        return

    provider = pending_payload["provider"]
    cfg = pending_payload["cfg"]
    messages_payload = pending_payload["messages"]
    temperature = float(pending_payload["temperature"])
    max_tokens = int(pending_payload["max_tokens"])

    radio_update = pending_payload["radio_update"]
    label_to_start = pending_payload["label_to_start"]
    ctx_html_string = pending_payload["ctx_html_string"]
    allowed_nums = set(pending_payload.get("allowed_nums", []))
    times_by_num = pending_payload.get("times_by_num", {})

    # Log size before streaming
    preview_text, total_chars, approx_tokens = messages_preview_text(messages_payload)
    _log_step("on_send_now:stream_start", provider=provider, total_chars=total_chars, approx_tokens=approx_tokens,
              temperature=temperature, max_tokens=max_tokens)

    stream = llm_chat_stream(
        provider=provider, cfg=cfg,
        messages=messages_payload,
        temperature=temperature, max_tokens=max_tokens
    )

    acc = ""
    msgs.append({"role": "assistant", "content": ""})
    for chunk in stream:
        if not chunk:
            continue
        acc += chunk
        msgs[-1]["content"] = acc
        yield (
            msgs,
            radio_update,
            label_to_start,
            ctx_html_string,
            gr.update(value="", visible=False),
            None,
            gr.update(visible=False),
        )

    msgs[-1]["content"] = sanitize_scene_output(acc, allowed_nums, times_by_num)
    _log_step("on_send_now:stream_end", total_chars=len(acc))
    yield (
        msgs,
        radio_update,
        label_to_start,
        ctx_html_string,
        gr.update(value="", visible=False),
        None,
        gr.update(visible=False),
    )


def hard_clear():
    _log_step("hard_clear")
    empty_msgs = []
    hide_radio = gr.update(choices=[], value=None, visible=False)
    empty_map = {}
    empty_ctx = ""
    hide_preview = gr.update(value="", visible=False)
    clear_pending = None
    clear_input = ""
    return empty_msgs, hide_radio, empty_map, empty_ctx, clear_input, hide_preview, clear_pending
