import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
MODELS={
    "bge":("Xenova/bge-small-en-v1.5","ea104dacec62c0de699686887e3f920caeb4f3e3",["tokenizer.json","onnx/model_quantized.onnx"]),
    "clip":("Xenova/clip-vit-base-patch32","6b7c523703ae26e39be4b03774966f8571c7f523",["tokenizer.json","preprocessor_config.json","onnx/text_model_quantized.onnx","onnx/vision_model_quantized.onnx"])
}
def get(item):
    group,repo,rev,name=item
    file=ROOT/"models"/group/Path(name).name
    file.parent.mkdir(parents=True,exist_ok=True)
    if not file.exists():
        temp=file.with_suffix(".part")
        subprocess.run(["curl","-fsSL","--retry","3","--connect-timeout","15","--max-time","600",f"https://huggingface.co/{repo}/resolve/{rev}/{name}","-o",str(temp)],check=True)
        temp.replace(file)
    print(file.name,file.stat().st_size,flush=True)

def get_qwen():
    from huggingface_hub import snapshot_download
    from qwen_retrieval import MODEL, REVISION, WEIGHT_HASH

    files = ["model.safetensors", "config.json", "chat_template.jinja", "vocab.json",
             "added_tokens.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
             "special_tokens_map.json", "preprocessor_config.json", "video_preprocessor_config.json"]
    snapshot_download("Qwen/Qwen3-VL-Embedding-2B", revision=REVISION, local_dir=MODEL, allow_patterns=files)
    with (MODEL / "model.safetensors").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHT_HASH:
            raise ValueError("Unexpected Qwen checkpoint checksum")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", action="store_true", help="Fetch the pinned Qwen embedding checkpoint")
    if parser.parse_args().qwen:
        get_qwen()
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(get,[(group,repo,rev,file) for group,(repo,rev,names) in MODELS.items() for file in names]))
