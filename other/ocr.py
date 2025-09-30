from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False, 
    use_doc_unwarping=False, 
    use_textline_orientation=False) # text detection + text recognition
# ocr = PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=True) # text image preprocessing + text detection + textline orientation classification + text recognition
# ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False) # text detection + textline orientation classification + text recognition
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_mobile_det",
#     text_recognition_model_name="PP-OCRv5_mobile_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False) # Switch to PP-OCRv5_mobile models
result = ocr.predict("./image.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")


def _load_paddleocr(use_gpu: bool) -> "PaddleOCR":
    if "ocr" not in _det_cache:
        if PaddleOCR is None:
            raise RuntimeError("No OCR backend found. Install paddleocr.")
        try:
            _det_cache["ocr"] = PaddleOCR(
                use_textline_orientation=True,  # replaces deprecated flag
                lang="en",
                use_gpu=use_gpu
            )
        except OSError:
            # If CUDA DLLs still fail, hard‑fallback to CPU
            _det_cache["ocr"] = PaddleOCR(
                use_textline_orientation=True, lang="en", use_gpu=False
            )
    return _det_cache["ocr"]