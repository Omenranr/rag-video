import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional
import shutil
import time
import datetime
import json
# from resources.consts import *
# from utils.common import *
# from utils.stt import *
# from utils.video import *
# from utils.chat import *
from resources.consts import *
from utils.visual import list_videos, find_existing_outputs_for_video, has_visual_artifacts, build_visual_index, ctx_md_from_hits_aggregated, hits_from_scene_rows, load_scene_rows_from_csv
from utils.stt import find_any_srt, find_audio_file_in, build_index_from_srt, srt_to_seconds
from utils.common import find_existing_indexes, safe_stem, load_index, format_source_label
from utils.chat import sanitize_scene_output, trim_history_messages, contextualize_question, format_history_as_text
from logic.intent_detection import detect_entity_intent, llm_decide_search, wants_recent_info, wants_web_search_explicit, llm_detect_intent_entities
from logic.entity_matching import llm_match_entities_to_keys, fuzzy_pick_entity
from logic.rag import compile_context_blocks_multi, auto_select_sources_from_query, load_entities_scenes_from_context
from chat.llm import llm_chat_stream
from tools.exa import exa_search_with_contents

# ---------------------
# Gradio callbacks
# ---------------------
def do_scan(folder):
    vids = list_videos(folder)
    dd_update = gr.update(choices=vids, value=(vids[0] if vids else None))
    vid_update = gr.update(value=None, visible=bool(vids))
    return dd_update, vid_update


# ---------------------
# Helper: toggle provider panels visibility
# ---------------------
def toggle_provider_panels(provider: str):
    p = (provider or "").lower()
    return (
        gr.update(visible=(p == "openai")),
        gr.update(visible=(p == "anthropic")),
    )


def on_select(video, outputs_root):
    vp_update = gr.update(value=video, visible=True)
    matches = find_existing_outputs_for_video(video, outputs_root)
    dd_update = gr.update(choices=matches, value=(matches[0] if matches else None))
    msg = "Found existing outputs:\n" + ("\n".join(matches[:8]) if matches else "None")
    return vp_update, dd_update, msg


def on_use_existing(selected_outputs, video_path, state_dict,
                    # transcript extractor controls
                    lang, vclass, fps, device, batch, conf, iou, max_det,
                    # visual controls
                    vlm_base_url, vlm_api_key, vlm_model, vlm_profile, vlm_maxconc, vlm_stream, vlm_stream_mode,
                    # index settings
                    window, anchor, embed_model, embed_device,
                    use_detection
                    ):
    if not selected_outputs:
        raise gr.Error("No outputs folder selected.")
    out_dir = Path(selected_outputs).resolve()
    if not out_dir.exists():
        raise gr.Error(f"Selected outputs folder doesn't exist: {out_dir}")

    # Detect existing artifacts
    # Probe what we already have
    srt_path = find_any_srt(str(out_dir))
    have_srt = srt_path is not None
    have_vlm = has_visual_artifacts(str(out_dir))
    existing_idx = find_existing_indexes(str(out_dir))
    idx_dir_trans = existing_idx.get("transcript")
    idx_dir_vlm   = existing_idx.get("visual")

    # CASE matrix from your requirement:
    # - none → generate SRT + VLM → build both indexes
    # - only SRT → run VLM → build both indexes (reuse SRT index if exists; else build)
    # - only VLM → run SRT → build both indexes (reuse VLM index if exists; else build)
    # - both → reuse; if any index missing, build it

    # === Transcript path / index logic ===
    # 1) If an index already exists for transcript, we can reuse it (no SRT required).
    # 2) Else, if any .srt exists, build the index from it.
    # 3) Else, if an audio file exists, transcribe from that audio (counts as "something present"),
    #    then build the index.
    # 4) Else, do not auto-run heavy jobs here; ask user to click Generate.

    if not idx_dir_trans:
        if not have_srt:
            audio_path = find_audio_file_in(str(out_dir))
            if audio_path is not None:
                # We DO run extractor here because the user already put an audio file in outputs.
                out_dir_str = run_extractor(
                    video_path=str(audio_path),
                    lang=lang, vclass=vclass, fps=(None if not fps else int(fps)),
                    device=device, batch_size=int(batch),
                    conf=float(conf), iou=float(iou), max_det=int(max_det),
                    progress=None,
                    enable_detection=bool(use_detection),   # <--- important
                )
                # The extractor might choose/return a different outputs folder
                if Path(out_dir_str).resolve() != out_dir:
                    out_dir = Path(out_dir_str).resolve()
                srt_path = find_any_srt(str(out_dir))
                have_srt = srt_path is not None
        # Build transcript index if we now have an SRT and no existing index
        if have_srt and not idx_dir_trans:
            idx_dir_trans = build_index_from_srt(
                transcript_path=str(srt_path),
                window=int(window), anchor=anchor,
                embed_model=embed_model,
                device=(None if embed_device == "auto" else embed_device),
                progress=None,
            )

    # === Visuals: only use if all 3 files exist; do NOT run extraction here ===
    # If artifacts exist and no visual index, build it. Otherwise leave it for "Generate".
    if have_vlm and not idx_dir_vlm:
        idx_dir_vlm = build_visual_index(
            outputs_dir=str(out_dir),
            embed_model=embed_model,
            embed_device=embed_device,
        )

    # Refresh existing indexes after possibly creating artifacts
    if not idx_dir_trans or not idx_dir_vlm:
        existing_idx = find_existing_indexes(str(out_dir))
        idx_dir_trans = idx_dir_trans or existing_idx.get("transcript")
        idx_dir_vlm   = idx_dir_vlm   or existing_idx.get("visual")

    # Cache/load
    if idx_dir_trans: _ = load_index(idx_dir_trans)
    if idx_dir_vlm:   _ = load_index(idx_dir_vlm)

    # Persist in state
    sd = state_dict or {}
    rec = sd.get(str(video_path), {"out_dir": str(out_dir), "index_dirs": {}})
    rec["out_dir"] = str(out_dir)
    if idx_dir_trans: rec["index_dirs"]["transcript"] = idx_dir_trans
    if idx_dir_vlm:   rec["index_dirs"]["visual"] = idx_dir_vlm
    sd[str(video_path)] = rec

    parts = [f"Using outputs: {out_dir}"]
    parts.append(f"- Transcript (.srt found): {'✔' if have_srt else '✖'}")
    parts.append(f"- Transcript index: {idx_dir_trans or '(none — click Generate if needed)'}")
    parts.append(f"- Visual artifacts (csv/context/meta): {'✔' if have_vlm else '✖'}")
    parts.append(f"- Visual index: {idx_dir_vlm or '(none — click Generate if needed)'}")
    if not have_srt and not idx_dir_trans:
        parts.append("• No transcript SRT or index found. If you didn’t drop an audio file in this folder, click “Generate”.")
    parts.append(f"- transcript index: {idx_dir_trans or '(none)'}")
    parts.append(f"- visual index:     {idx_dir_vlm   or '(none)'}")
    return "\n".join(parts), sd


def do_generate(
    folder, video_path, outputs_root,
    lang, vclass, fps, device, batch, conf, iou, max_det,
    window, anchor, embed_model, embed_device, state_dict,
    vlm_base_url, vlm_api_key, vlm_model, vlm_profile, vlm_maxconc, vlm_stream, vlm_stream_mode,
    use_transcript: bool, use_detection: bool, use_visual: bool,   # <-- order aligned with new UI
    progress=gr.Progress(track_tqdm=True)):

    if not video_path:
        raise gr.Error("Please select a video.")
    progress(0.0, desc="Starting…")

    need_extractor = bool(use_transcript or use_detection)
    out_dir: Optional[str] = None
    idx_dir_trans = None
    idx_dir_vlm = None

    # Decide or create outputs dir
    existing = find_existing_outputs_for_video(video_path, outputs_root)
    out_dir = existing[0] if existing else None
    if not out_dir:
        stem = safe_stem(Path(video_path))
        out_dir = str(Path(outputs_root).expanduser() / f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Run extractor only if SRT or detection is requested
    if need_extractor:
        srt_pre = find_any_srt(out_dir)
        # We still run extractor if detection is enabled (even if SRT already exists), so we can generate detections.* files
        must_run = (use_detection or not srt_pre)  # run if we need detections OR we need a transcript
        if must_run:
            out_dir = run_extractor(
                video_path=video_path, lang=lang, vclass=vclass, fps=(None if not fps else int(fps)),
                device=device, batch_size=int(batch), conf=float(conf), iou=float(iou), max_det=int(max_det),
                progress=progress, enable_detection=use_detection
            )
            progress(0.4, desc="Extractor finished.")

    # Build transcript index only if transcript requested and exists/was created
    if use_transcript:
        srt_path = find_any_srt(str(out_dir))
        if not srt_path:
            raise gr.Error("Transcript requested but no .srt was produced/found.")
        idx_dir_trans = build_index_from_srt(
            transcript_path=str(srt_path),
            window=int(window), anchor=anchor,
            embed_model=embed_model,
            device=(None if embed_device == "auto" else embed_device),
            progress=progress,
        )
        progress(0.7, desc="Transcript index ready.")

    # VLM pipeline (only if requested)
    if use_visual:
        have_vlm = has_visual_artifacts(out_dir)
        if not have_vlm:
            progress(0.75, desc="Visual extraction…")
            run_visual_strategy(
                video_path=video_path, outputs_dir=out_dir,
                base_url=vlm_base_url.strip(), api_key=vlm_api_key.strip(),
                model=vlm_model.strip(), profile=vlm_profile,
                max_concurrency=int(vlm_maxconc),
                stream=bool(vlm_stream), stream_mode=vlm_stream_mode,
                force=False,
            )
            progress(0.84, desc="Visual artifacts generated.")
        idx_dir_vlm = build_visual_index(
            outputs_dir=out_dir,
            embed_model=embed_model,
            embed_device=embed_device,
        )
        progress(0.9, desc="Visual index ready.")

    # Save to state
    sd = state_dict or {}
    rec = {"out_dir": out_dir, "index_dirs": {}}
    if idx_dir_trans: rec["index_dirs"]["transcript"] = idx_dir_trans
    if idx_dir_vlm:   rec["index_dirs"]["visual"] = idx_dir_vlm
    sd[str(video_path)] = rec
    progress(1.0, desc="Done.")

    status = [f"✅ Prepared in: {out_dir}"]
    status.append(f"- Transcript index: {idx_dir_trans or '(skipped)'}")
    status.append(f"- Visual index:     {idx_dir_vlm or '(skipped)'}")
    status.append(f"- Object detection: {'enabled' if use_detection else 'disabled'}")
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
    source_mode
):
    """
    Generator that yields (messages, ts_radio_update, ts_map_state, ctx_html_string).
    Chatbot(type='messages'): messages must be [{'role','content'}, ...]
    Now **history-aware**: we use chat history to contextualize the query and in the final answer.
    """
    # The chat as displayed to the user
    msgs = list(history or [])

    # basic validations
    if not video_path:
        msgs.append({"role": "assistant", "content": "Please select a video first."})
        yield msgs, gr.update(choices=[], value=None, visible=False), {}, ""
        return

    cfg = _provider_cfg(provider, oa_base_url, oa_api_key, oa_model,
                        an_api_key, an_model)
    err = _validate_provider_inputs(provider, cfg)
    if err:
        msgs.append({"role": "assistant", "content": err})
        yield msgs, gr.update(choices=[], value=None, visible=False), {}, ""
        return

    rec = (state_dict or {}).get(str(video_path))

    if not rec:
        msgs.append({"role": "assistant", "content": "No outputs mapped for this video. Click 'Use selected outputs' or 'Generate'."})
        yield msgs, gr.update(choices=[], value=None, visible=False), {}, ""
        return

    # Append the user's latest message to the visible chat
    latest_user = str(user_msg)
    if not (msgs and msgs[-1].get("role") == "user" and msgs[-1].get("content") == latest_user):
        msgs.append({"role": "user", "content": latest_user})

    # Prepare a trimmed history window (EXCLUDES the latest user msg for clarity in prompts)
    history_window_for_rewrite = trim_history_messages(msgs[:-1] or [], max_turns=10, max_chars=6000)
    history_text_for_router = format_history_as_text(msgs[:-1] or [], max_turns=10, max_chars=6000)

    # === (1) Contextualize the question with history ===
    try:
        standalone_q = contextualize_question(
            provider=(provider or "anthropic"),
            cfg=cfg,
            history_msgs=history_window_for_rewrite,  # only prior messages
            latest_user_msg=latest_user,
        )
    except Exception:
        standalone_q = latest_user

    # === (2a) Entity-search intention? Prefer entities_scenes over RAG ===
    is_entity, entity_name = detect_entity_intent(standalone_q)
    if is_entity and entity_name:
        # Load entities_scenes from *_context.json in this video's outputs
        out_dir = (state_dict or {}).get(str(video_path), {}).get("out_dir")
        entities_scenes, norm_map = load_entities_scenes_from_context(out_dir) if out_dir else ({}, {})
        resolved_key = fuzzy_pick_entity(entity_name, norm_map)

        if entities_scenes and resolved_key in entities_scenes:
            # Found: collect target scenes → build context from CSV rows
            scene_ids = sorted({int(s) for s in entities_scenes.get(resolved_key, [])})
            scene_rows = load_scene_rows_from_csv(out_dir, scene_ids)
            hits = hits_from_scene_rows(scene_rows)
            ctx_text = "\n\n".join(
                [f"[Scene {r['scene_id']}] {r['start_timecode']}–{r['end_timecode']}\n{r['generated_text']}" for r in scene_rows]
            )
            ctx_html_string = ctx_md_from_hits_aggregated(hits, title=f"Scenes for entity: {resolved_key}")

            # Prepare timecode radio options
            labels = []
            label_to_start = {}
            for h in hits:
                c0 = h["context"][0]
                snippet = (c0.get("text","") or "").replace("\n"," ")
                if len(snippet) > 100: snippet = snippet[:97] + "..."
                label = f"[VLM] {c0['start_srt']} - {c0['end_srt']} | {snippet}"
                labels.append(label)
                label_to_start[label] = srt_to_seconds(c0["start_srt"])
            radio_update = gr.update(choices=labels, value=None, visible=bool(labels))

            # Compose final answer with LLM (grounded in these scenes)
            system = (
                "Tu es un assistant vidéo. Règles STRICTES:\n"
                "• Pour chaque élément, préfixe le timecode EXACT avec **SCENE N°{num}: {HH:MM:SS,mmm–HH:MM:SS,mmm}**.\n"
                "• {num} = scene_id s'il est fourni, sinon le numéro du passage (rank).\n"
                "• 1–2 lignes par scène. N'invente rien au-delà du contexte fourni."
                f"ALLOWED_SCENES = {json.dumps([{'num': int(r['scene_id']), 'start': r['start_timecode'], 'end': r['end_timecode'], 'src': 'VLM'} for r in scene_rows], ensure_ascii=False)}"
            )

            user_full = (
                f"Requête utilisateur: Donne toutes les scènes où «{entity_name}» apparaît.\n"
                f"Entité résolue (normalisée): {resolved_key}\n"
                f"Scenes trouvées: {scene_ids}\n\n"
                f"Contexte par scène (cartes VLM):\n{ctx_text}\n\n"
                "Tâche: Résume et liste les scènes sous forme de puces: [scene_id] HH:MM:SS,mmm–HH:MM:SS,mmm — 1 à 2 lignes utiles."
            )

            stream = llm_chat_stream(
                provider=provider, cfg=cfg,
                messages=[{"role":"system","content":system},{"role":"user","content":user_full}],
                temperature=0.1, max_tokens=2000
            )

            for out in _yield_stream_with_sanitize(
                stream,
                msgs,
                radio_update,
                label_to_start,
                ctx_html_string,
                allowed_nums,
                times_by_num
            ):
                yield out
            return

    # === (2a) LLM: detect intention & extract entities for entity_search ===
    try:
        intent_res = llm_detect_intent_entities(
            provider=(provider or "anthropic"),
            cfg=cfg,
            latest_user_msg=str(standalone_q),
            history_msgs=(history or []),
        )
    except Exception:
        intent_res = {"intent":"other","entities":[],"reason":"router failed"}

    if intent_res.get("intent") == "entity_search" and intent_res.get("entities"):
        # Load entities_scenes and delegate matching to the LLM
        out_dir = (state_dict or {}).get(str(video_path), {}).get("out_dir")
        entities_scenes, candidate_keys = load_entities_scenes_from_context(out_dir) if out_dir else ({}, [])
        if entities_scenes and candidate_keys:
            try:
                match_res = llm_match_entities_to_keys(
                    provider=(provider or "anthropic"),
                    cfg=cfg,
                    query_entities=intent_res["entities"],
                    candidate_keys=candidate_keys,
                )
            except Exception:
                match_res = {"matches": [], "flat_keys": []}

            matched_keys: List[str] = match_res.get("flat_keys") or []
            if matched_keys:
                # Union all scene IDs for the selected keys
                scene_ids = sorted({int(s) for k in matched_keys for s in (entities_scenes.get(k) or [])})
                scene_rows = load_scene_rows_from_csv(out_dir, scene_ids)
                hits = hits_from_scene_rows(scene_rows)
                ctx_text = "\n\n".join(
                    [f"[Scene {r['scene_id']}] {r['start_timecode']}–{r['end_timecode']}\n{r['generated_text']}" for r in scene_rows]
                )
                ctx_html_string = ctx_md_from_hits_aggregated(hits, title=f"Scenes for entities: {', '.join(matched_keys)}")

                # Prepare the timecode radio
                labels = []
                label_to_start = {}
                for h in hits:
                    c0 = h["context"][0]
                    snippet = (c0.get("text","") or "").replace("\n"," ")
                    if len(snippet) > 100: snippet = snippet[:97] + "..."
                    label = f"[VLM] {c0['start_srt']} - {c0['end_srt']} | {snippet}"
                    labels.append(label)
                    label_to_start[label] = srt_to_seconds(c0["start_srt"])
                radio_update = gr.update(choices=labels, value=None, visible=bool(labels))

                # Build allow-list from the VLM scene rows
                allowed = []
                for r in scene_rows:
                    num = int(r["scene_id"])
                    allowed.append({
                        "num": num,
                        "start": r["start_timecode"],
                        "end": r["end_timecode"],
                        "src": "VLM",
                    })
                allowed_nums = {a["num"] for a in allowed}
                times_by_num = {a["num"]: (a["start"], a["end"]) for a in allowed}

                # Final answer grounded on those scenes
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

                try:
                    stream = llm_chat_stream(
                        provider=(provider or "anthropic"),
                        cfg=cfg,
                        messages=[{"role":"system","content":system},{"role":"user","content":user_full}],
                        temperature=0.1,
                        max_tokens=700,
                    )
                    for out in _yield_stream_with_sanitize(
                        stream,
                        msgs,
                        radio_update,
                        label_to_start,
                        ctx_html_string,
                        allowed_nums,
                        times_by_num
                    ):
                        yield out
                    return
                except Exception as e:
                    msgs.append({"role":"assistant","content": f"LLM error: {e}"})
                    yield msgs, radio_update, label_to_start, ctx_html_string
                    return

    # === (2) Decide which indexes to use ===
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
        # AUTO: classify using the rewritten standalone question
        want_transcript, want_visual, _reason = auto_select_sources_from_query(standalone_q)

    # Fallbacks if selection not available
    if want_transcript and not have_trans:
        want_transcript = False
    if want_visual and not have_visual:
        want_visual = False
    if not want_transcript and not want_visual:
        # If nothing matches availability, fall back to whichever exists
        if have_trans:      want_transcript = True
        elif have_visual:   want_visual = True
        else:
            msgs.append({"role": "assistant", "content": "No RAG indexes are available for this video. Generate or load indexes first."})
            yield msgs, gr.update(choices=[], value=None, visible=False), {}, ""
            return

    idx_pairs = []
    if want_transcript and have_trans:
        idx_pairs.append((load_index(available["transcript"]), "SRT"))
    if want_visual and have_visual:
        idx_pairs.append((load_index(available["visual"]), "VLM"))

    hits, ctx_text = compile_context_blocks_multi(
        indexes=idx_pairs,
        query=str(standalone_q), top_k=int(top_k), method=method, alpha=float(alpha),
        rerank=rerank, rerank_model=rerank_model, overfetch=int(overfetch),
        ctx_before=int(ctx_before), ctx_after=int(ctx_after),
        device=(None if embed_device == "auto" else embed_device),
        embed_model_override=(None if not embed_model_override else embed_model_override)
    )

    # Build an allow-list of scene ids and times for the current answer turn
    allowed = []
    for h in hits:
        # prefer a real scene_id (VLM) else fall back to 'rank' as the scene number
        num = int(h.get("scene_id", h.get("rank", 0)) or h.get("rank", 0))
        ctx_sorted = sorted(h["context"], key=lambda c: (c.get("offset",0)!=0, c.get("offset",0)))
        main = next((c for c in ctx_sorted if c.get("offset",0)==0), ctx_sorted[0])
        allowed.append({
            "num": num,
            "start": main.get("start_srt","00:00:00,000"),
            "end":   main.get("end_srt","00:00:00,000"),
            "src":   h.get("source","srt").upper(),  # SRT/VLM
        })

    # Convenience maps for post-validation
    allowed_nums = {a["num"] for a in allowed}
    times_by_num = {a["num"]: (a["start"], a["end"]) for a in allowed}

    ctx_html_string = ctx_md_from_hits_aggregated(hits, title="Retrieved passages")

    # Prepare timecode radio options now (from hits), to return with any yield
    labels = []
    label_to_start = {}
    for h in hits:
        ctx_sorted = sorted(h["context"], key=lambda c: (c["offset"] != 0, c["offset"]))
        main = next((c for c in ctx_sorted if c.get("offset", 0) == 0), ctx_sorted[0])
        start_srt = main.get("start_srt", h.get("start_srt", "00:00:00,000"))
        end_srt = main.get("end_srt", h.get("end_srt", "00:00:00,000"))
        snippet = (main.get("text", "") or "").replace("\n", " ")
        if len(snippet) > 100:
            snippet = snippet[:97] + "..."
        source_label = format_source_label(h.get("source"))
        label = f"[{source_label}] {start_srt} - {end_srt} | {snippet}".strip()
        labels.append(label)
        label_to_start[label] = srt_to_seconds(start_srt)

    radio_update = gr.update(choices=labels, value=None, visible=bool(labels))

    # === (3) Router: decide about web search (history-aware) ===
    explicit_flag = wants_web_search_explicit(latest_user)

    # NEW: recency detection on both the raw and rewritten question
    cy = datetime.now().year
    rec_flag1, yrs1 = wants_recent_info(latest_user, cy)
    rec_flag2, yrs2 = wants_recent_info(standalone_q, cy)
    recency_flag = rec_flag1 or rec_flag2
    years_mentioned = sorted(set(yrs1 + yrs2))

    if recency_flag and (not enable_web or not exa_api_key):
        msgs.append({"role":"assistant",
                    "content":"This looks time-sensitive (e.g., 'now/2025'). Enable web search (Exa) to fetch up-to-date info, otherwise I can only answer from the video’s content."})
        msgs[-1]["content"] = sanitize_scene_output(msgs[-1]["content"], allowed_nums, times_by_num)
        yield msgs, radio_update, label_to_start, ctx_html_string
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
    except Exception as e:
        msgs.append({"role": "assistant", "content": f"Routing error: {e}"})
        msgs[-1]["content"] = sanitize_scene_output(msgs[-1]["content"], allowed_nums, times_by_num)
        yield msgs, radio_update, label_to_start, ctx_html_string
        return

    need_search = bool(decision.get("need_search"))
    search_query = decision.get("query")
    direct_answer = decision.get("answer")

    # === (4A) If no web search needed, answer using transcript + history ===
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
        try:
            stream = llm_chat_stream(
                provider=(provider or "anthropic"),
                cfg=cfg,
                messages=[{"role": "system", "content": system}, *history_for_model, {"role": "user", "content": user_full}],
                temperature=0.2,
                max_tokens=900,
            )
            for out in _yield_stream_with_sanitize(
                stream, msgs, radio_update, label_to_start, ctx_html_string, allowed_nums, times_by_num
            ):
                yield out
            return
        except Exception as e:
            msgs.append({"role": "assistant", "content": f"LLM error: {e}"})
            yield msgs, radio_update, label_to_start, ctx_html_string
            return


    # === (4B) Web search suggested ===
    if not search_query:
        msgs.append({"role": "assistant", "content": "La recherche web a été suggérée, mais aucune requête n'a été fournie."})
        yield msgs, radio_update, label_to_start, ctx_html_string
        return

    if not enable_web:
        msgs.append({"role": "assistant", "content": f"🔎 Requête web suggérée : \"{search_query}\" (la recherche web est désactivée)."})
        yield msgs, radio_update, label_to_start, ctx_html_string
        return

    if not exa_api_key:
        msgs.append({"role": "assistant", "content": f"🔎 Requête web suggérée : \"{search_query}\" (ajoutez une clé Exa pour effectuer la recherche)."})
        yield msgs, radio_update, label_to_start, ctx_html_string
        return

    # Show interim "searching" message
    msgs.append({"role": "assistant", "content": f"🔎 Web search query: \"{search_query}\" (running Exa…)"})
    yield msgs, radio_update, label_to_start, ctx_html_string

    # 3) Exa search
    try:
        web_hits = exa_search_with_contents(search_query, exa_api_key, num_results=int(exa_num_results))
    except Exception as e:
        msgs.append({"role": "assistant", "content": f"Exa search failed: {e}"})
        yield msgs, radio_update, label_to_start, ctx_html_string
        return

    if not web_hits:
        msgs.append({"role": "assistant", "content": "Aucun résultat web pertinent n'a été trouvé."})
        yield msgs, radio_update, label_to_start, ctx_html_string
        return

    # Build web context blob
    web_blocks = []
    for i, w in enumerate(web_hits, start=1):
        block = f"[{i}] {w['title']}  {w['url']}\n{w['snippet']}"
        web_blocks.append(block)
    web_context = "\n\n".join(web_blocks)

    # 4) Final LLM call with history + transcript + web context
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
    try:
        stream = llm_chat_stream(
            provider=(provider or "anthropic"),
            cfg=cfg,
            messages=[{"role": "system", "content": system}, *history_for_model, {"role": "user", "content": user_full}],
            temperature=0.2,
            max_tokens=900,
        )
        for out in _yield_stream_with_sanitize(
            stream,
            msgs,
            radio_update,
            label_to_start,
            ctx_html_string,
            allowed_nums,
            times_by_num
        ):
            yield out
        return
    except Exception as e:
        msgs.append({"role": "assistant", "content": f"LLM error (final): {e}"})
        yield msgs, radio_update, label_to_start, ctx_html_string
        return


def on_upload_video(file_path: str, videos_dir: str):
    if not file_path:
        raise gr.Error("No file uploaded.")

    src = Path(file_path)
    if not src.exists():
        raise gr.Error("Upload failed: temporary file not found.")

    ext = src.suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise gr.Error(f"Invalid file type: {ext}. Allowed: {sorted(ALLOWED_VIDEO_EXTS)}")

    dst_dir = Path(videos_dir).expanduser()
    dst_dir.mkdir(parents=True, exist_ok=True)

    # unique name if collision
    dest = dst_dir / src.name
    if dest.exists():
        dest = dst_dir / f"{src.stem}_{int(time.time())}{src.suffix}"

    shutil.copy2(src, dest)

    # refresh list
    vids = list_videos(str(dst_dir))  # you already have this helper
    return (
        gr.update(choices=vids, value=str(dest)),           # dropdown
        gr.update(value=str(dest), visible=True),           # video player
        f"✅ Uploaded to: {dest}"                            # status
    )


def hard_clear():
    # Clears visible chat + related UI/state you use during a turn
    empty_msgs = []
    hide_radio = gr.update(choices=[], value=None, visible=False)
    empty_map = {}
    empty_ctx = ""   # context panel
    return empty_msgs, hide_radio, empty_map, empty_ctx, ""
