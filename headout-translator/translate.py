import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class HeadoutTranslator:
    def __init__(self, model_dir: str, device: str = None):
        cfg = json.load(open(Path(model_dir) / "inference_config.json"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.src_lang = cfg["src_lang"]
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16
        ).to(self.device).eval()
        self.cfg = cfg

    def translate(self, text: str) -> str:
        inputs = self.tok(text, return_tensors="pt",
                          max_length=self.cfg["max_new_tokens"], truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=self.cfg["fra_token_id"],
                num_beams=self.cfg["num_beams"],
                length_penalty=self.cfg["length_penalty"],
                max_new_tokens=self.cfg["max_new_tokens"],
            )
        return self.tok.decode(out[0], skip_special_tokens=True)

    def translate_batch(self, texts: list[str]) -> list[str]:
        inputs = self.tok(texts, return_tensors="pt", padding=True,
                          max_length=self.cfg["max_new_tokens"], truncation=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=self.cfg["fra_token_id"],
                num_beams=self.cfg["num_beams"],
                length_penalty=self.cfg["length_penalty"],
                max_new_tokens=self.cfg["max_new_tokens"],
            )
        return self.tok.batch_decode(out, skip_special_tokens=True)

if __name__ == "__main__":
    t = HeadoutTranslator("./headout-translator")
    print(t.translate("Skip the Line tickets for the Eiffel Tower."))
