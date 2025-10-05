import gradio as gr
from pathlib import Path
import os
from ui.callbacks import do_scan, hard_clear, on_chat, do_generate, on_use_existing, on_upload_video, on_select, toggle_provider_panels, on_send_now
from dotenv import load_dotenv

load_dotenv()


# ---------------------
# Gradio Layout
# ---------------------
with gr.Blocks(title="Agentic Video RAG Chat", fill_height=True) as demo:
    gr.Markdown("## Agentic Video RAG Chat\nScan → select video → reuse **existing outputs** or **Generate** → chat with transcript.\nOptional: let the assistant trigger **web search** via Exa when needed.\n\n**LLM Providers:** OpenAI • Anthropic\n\n**New:** answers use **RAG + chat history** for follow-ups and pronouns.")

    with gr.Row():
        with gr.Column(scale=3):
            upload_video = gr.File(
                label="Upload a video",
                file_count="single",
                file_types=["video"],
                type="filepath"
            )

            gr.HTML("""
            <style>
            /* Hide the built-in Clear button on the Chatbot header */
            #chat_box button[aria-label="Clear"] { display: none !important; }
            </style>
            """)

            folder_tb = gr.Textbox(label="Folder to scan for videos", value=str(Path.cwd() / "videos"))
            outputs_root_tb = gr.Textbox(label="Outputs folder", value=str(Path.cwd() / "outputs"))
            scan_btn = gr.Button("Scan videos")

            video_dd = gr.Dropdown(choices=[], label="Select a video")
            existing_outputs_dd = gr.Dropdown(choices=[], label="Existing outputs for this video")
            use_existing_btn = gr.Button("Use selected outputs")

            video_player = gr.Video(label="Preview", elem_id="video_preview")
            status_box = gr.Textbox(label="Status", lines=8)

            with gr.Accordion("Pipelines to run", open=True):
                use_srt_chk = gr.Checkbox(value=True, label="Transcript (SRT)")
                use_detect_chk = gr.Checkbox(value=False, label="Object detection (YOLO, etc.)")
                use_vlm_chk = gr.Checkbox(value=True, label="VLM visual cards")

            with gr.Accordion("Extraction settings (used only when you click Generate)", open=False):
                lang_dd = gr.Dropdown(choices=["EN","FR"], value="FR", label="Transcript language (Wit.ai key must exist)")
                vclass_dd = gr.Dropdown(choices=["sport","nature","screen","cctv"], value="sport", label="Detection pipeline")
                fps_tb = gr.Number(value=None, label="Override FPS (optional)")
                device_dd = gr.Dropdown(choices=["auto","cpu","cuda"], value="auto", label="Detector device")
                batch_tb = gr.Slider(8, 128, 64, step=8, label="Batch size")
                conf_tb = gr.Slider(0.05, 0.9, 0.25, step=0.05, label="Det conf")
                iou_tb = gr.Slider(0.1, 0.95, 0.7, step=0.05, label="NMS IoU")
                maxdet_tb = gr.Slider(50, 1000, 300, step=50, label="Max det/frame")

            with gr.Accordion("Visual extraction", open=False):
                vlm_base_url_tb = gr.Textbox(label="vLLM Base URL", value=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
                vlm_api_key_tb = gr.Textbox(label="vLLM API Key", type="password", value=os.getenv("VLLM_API_KEY", "apikey-1234"))
                vlm_model_tb = gr.Textbox(label="VLM Model", value=os.getenv("VLLM_MODEL", "OpenGVLab/InternVL3-8B"))
                vlm_profile_dd = gr.Dropdown(choices=["brand_heavy","sports","compliance","slides_docs","default"], value="brand_heavy", label="Profile")
                vlm_maxconc_tb = gr.Slider(2, 20, 8, step=1, label="Max concurrency")
                vlm_stream_chk = gr.Checkbox(value=False, label="Stream live to console (server logs)")
                vlm_stream_mode_dd = gr.Dropdown(choices=["aggregate","sequential","none"], value="none", label="Stream mode")

            with gr.Accordion("Index sources to use in Chat", open=True):
                source_mode_dd = gr.Dropdown(
                    choices=["auto", "both", "transcript", "visual"],
                    value="auto",
                    label="Index source mode"
                )


            with gr.Accordion("Index settings (also used if existing outputs lack an index)", open=False):
                window_tb = gr.Slider(3, 15, 10, step=1, label="Chunk window (seconds)")
                anchor_dd = gr.Dropdown(choices=["first","zero"], value="first", label="Window anchor")
                embed_model_tb = gr.Textbox(value="sentence-transformers/all-MiniLM-L6-v2", label="Embedding model")
                embed_device_dd = gr.Dropdown(choices=["auto","cpu","cuda"], value="auto", label="Embed device")

            with gr.Accordion("Web Search (Exa)", open=False):
                enable_web_chk = gr.Checkbox(value=True, label="Enable web search with Exa")
                exa_key_tb = gr.Textbox(label="Exa API Key", type="password", value=os.getenv("EXA_API_KEY", ""))
                exa_num_tb = gr.Slider(1, 12, 5, step=1, label="Exa: number of results")

            generate_btn = gr.Button("🧪 Generate (transcript + detections + index)", variant="primary")

        with gr.Column(scale=4):
            with gr.Tab("Chat"):
                with gr.Accordion("Retrieval & Rerank", open=False):
                    ctx_before_tb = gr.Slider(0, 6, 1, step=1, label="Context: previous chunks")
                    ctx_after_tb = gr.Slider(0, 6, 1, step=1, label="Context: following chunks")
                    topk_tb = gr.Slider(1, 20, 6, step=1, label="Top-K chunks")
                    method_dd = gr.Dropdown(choices=["rrf","weighted","bm25","embed"], value="rrf", label="Base retrieval")
                    alpha_tb = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="Weighted α (embed weight)")
                    rerank_dd = gr.Dropdown(choices=["none","cross","mmr"], value="none", label="Rerank")
                    rerank_model_tb = gr.Textbox(value="cross-encoder/ms-marco-MiniLM-L-6-v2", label="Cross-encoder model")
                    overfetch_tb = gr.Slider(10, 200, 50, step=10, label="Overfetch for rerank")
                    embed_model_override_tb = gr.Textbox(value="", label="Override embed model at query time (optional)")
                    embed_device_q_dd = gr.Dropdown(choices=["auto","cpu","cuda"], value="auto", label="Embed device (query)")

                with gr.Accordion("LLM Provider & connection", open=True):
                    provider_dd = gr.Dropdown(choices=["openai","anthropic"], value="anthropic", label="Provider")

                    with gr.Group(visible=False) as oa_group:
                        gr.Markdown("#### OpenAI (Chat Completions)")
                        oa_base_url_tb = gr.Textbox(label="OpenAI Base URL (optional)", placeholder="https://api.openai.com")
                        oa_api_key_tb = gr.Textbox(label="OpenAI API Key", type="password")
                        oa_model_tb = gr.Textbox(label="OpenAI Model", placeholder="gpt-4o-mini or gpt-4o")

                    with gr.Group(visible=True) as an_group:
                        gr.Markdown("#### Anthropic (Messages API)")
                        an_api_key_tb = gr.Textbox(label="Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
                        an_model_tb = gr.Textbox(label="Anthropic Model", value="claude-3-5-haiku-20241022")

                with gr.Accordion("Timecodes (click to seek)", open=False):
                    ts_radio = gr.Radio(choices=[], label=None, interactive=True, visible=False)

                # Collapsible panel with the full retrieved context (all passages + windows)
                with gr.Accordion("Retrieved context (expand to view)", open=False):
                    ctx_panel = gr.HTML(value="", elem_id="ctx_md_full")

                ts_map_state = gr.State({})  # label -> start_seconds

                chat = gr.Chatbot(height=520, type="messages", elem_id="chat_box")
                chat_tb = gr.Textbox(placeholder="Ask about the video…", label="Message")
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("🗑️ Clear chat (hard)")
                with gr.Accordion("Dispatch mode", open=True):
                    dispatch_mode_dd = gr.Dropdown(
                        choices=["Auto send", "Check before"],
                        value="Auto send",
                        label="LLM dispatch"
                    )

                with gr.Accordion("LLM payload preview (only in 'Check before')", open=False) as preview_section:
                    preview_html = gr.HTML(value="", elem_id="llm_preview", visible=False)
                    send_now_btn = gr.Button("Send to LLM now", variant="primary", visible=False)

                pending_payload_state = gr.State(None)   # holds {provider, cfg, messages, temp, max_tokens, ui carry-ons}

            state_dict = gr.State({})

    # Wiring
    scan_btn.click(fn=do_scan, inputs=[folder_tb], outputs=[video_dd, video_player])
    video_dd.change(fn=on_select, inputs=[video_dd, outputs_root_tb], outputs=[video_player, existing_outputs_dd, status_box])

    provider_dd.change(toggle_provider_panels, inputs=[provider_dd], outputs=[oa_group, an_group])


    use_existing_btn.click(
        fn=on_use_existing,
        inputs=[
            existing_outputs_dd, video_dd, state_dict,
            # transcript extractor controls
            lang_dd, vclass_dd, fps_tb, device_dd, batch_tb, conf_tb, iou_tb, maxdet_tb,
            # visual controls
            vlm_base_url_tb, vlm_api_key_tb, vlm_model_tb, vlm_profile_dd, vlm_maxconc_tb, vlm_stream_chk, vlm_stream_mode_dd,
            # index settings
            window_tb, anchor_dd, embed_model_tb, embed_device_dd,
            use_detect_chk
        ],
        outputs=[status_box, state_dict]
    )

    upload_video.upload(
        fn=on_upload_video,
        inputs=[upload_video, folder_tb],
        outputs=[video_dd, video_player, status_box]
    )

    generate_btn.click(
        fn=do_generate,
        inputs=[
            folder_tb, video_dd, outputs_root_tb,
            lang_dd, vclass_dd, fps_tb, device_dd, batch_tb, conf_tb, iou_tb, maxdet_tb,
            window_tb, anchor_dd, embed_model_tb, embed_device_dd, state_dict,
            vlm_base_url_tb, vlm_api_key_tb, vlm_model_tb, vlm_profile_dd, vlm_maxconc_tb, vlm_stream_chk, vlm_stream_mode_dd,
            use_srt_chk, use_detect_chk, use_vlm_chk,   # <-- NEW
        ],
        outputs=[status_box, state_dict]
    )

    # Click a timecode → seek the video (JS)
    ts_radio.change(
        inputs=[ts_radio, ts_map_state],
        outputs=[status_box],   # debug info; remove if you don't want logs
        js="""
        (label, map) => {
        const logs = [];
        logs.push(`label: ${label}`);

        const parseFromLabel = (lbl) => {
            const m = /\b(\d{2}):(\d{2}):(\d{2}),(\d{3})\b/.exec(lbl || "");
            if (!m) return NaN;
            const hh = +m[1], mm = +m[2], ss = +m[3];
            return hh*3600 + mm*60 + ss;
        };
        let seconds = (map && typeof map[label] === "number") ? map[label] : parseFromLabel(label);
        logs.push(`seconds: ${seconds}`);
        if (!Number.isFinite(seconds)) return `Bad seconds from label/map`;

        const host = document.querySelector("#video_preview");
        logs.push(`host found: ${!!host}`);

        const tryFindVideo = (root) => {
            if (!root) return null;
            let v = root.querySelector ? root.querySelector("video") : null;
            if (v) return v;
            const gv = root.querySelector ? root.querySelector("gradio-video") : null;
            if (gv && gv.shadowRoot) {
                v = gv.shadowRoot.querySelector("video");
                if (v) return v;
            }
            const all = root.querySelectorAll ? root.querySelectorAll("*") : [];
            for (const el of all) {
                if (el.shadowRoot) {
                    const vv = el.shadowRoot.querySelector("video");
                    if (vv) return vv;
                }
            }
            return null;
        };

        let video = tryFindVideo(host);
        if (!video) {
            const gvAll = Array.from(document.querySelectorAll("gradio-video"));
            for (const gv of gvAll) {
                if (gv.shadowRoot) {
                    const v2 = gv.shadowRoot.querySelector("video");
                    if (v2) { video = v2; break; }
                }
            }
        }
        logs.push(`video found: ${!!video}`);
        if (!video) return `No <video> element found`;

        const seek = () => {
            try {
                const dur = Number.isFinite(video.duration) ? video.duration : Infinity;
                const t = Math.max(0, Math.min(dur, seconds));
                video.currentTime = t;
                if (video.paused) { video.play().catch(()=>{}); }
                logs.push(`seeked to ${t}s`);
            } catch (e) {
                logs.push(`seek error: ${e}`);
            }
        };

        if (video.readyState >= 1) seek();
        else video.addEventListener("loadedmetadata", seek, { once: true });

        return logs.join("\\n");
        }
        """
    )

    # Chat callback (messages + timecode radio)
    def _chat_send(
        user_msg, chat_history,
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
        for out in on_chat(
            user_msg, chat_history,
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
            yield out

    send_btn.click(
        _chat_send,
        inputs=[
            chat_tb, chat,
            video_dd, ctx_before_tb, ctx_after_tb, topk_tb, method_dd, alpha_tb,
            rerank_dd, rerank_model_tb, overfetch_tb,
            provider_dd,
            oa_base_url_tb, oa_api_key_tb, oa_model_tb,
            an_api_key_tb, an_model_tb,
            embed_device_q_dd, embed_model_override_tb,
            enable_web_chk, exa_key_tb, exa_num_tb,
            state_dict,
            source_mode_dd,
            dispatch_mode_dd,
        ],
        outputs=[
            chat,                     # messages
            ts_radio,                 # radio
            ts_map_state,             # label -> start
            ctx_panel,                # context html
            preview_html,             # NEW
            pending_payload_state,    # NEW
            send_now_btn,             # NEW (button visibility)
        ]
    ).then(lambda: "", None, [chat_tb])

    clear_btn.click(
        hard_clear,
        inputs=None,
        outputs=[chat, ts_radio, ts_map_state, ctx_panel, chat_tb, preview_html, pending_payload_state]
    )
    send_now_btn.click(
        on_send_now,
        inputs=[chat, pending_payload_state],
        outputs=[
            chat,
            ts_radio,
            ts_map_state,
            ctx_panel,
            preview_html,
            pending_payload_state,
            send_now_btn,
        ]
    )
