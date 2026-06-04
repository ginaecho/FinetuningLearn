"""
Download / verify access to a gated Hugging Face model (default: Meta-Llama-3-8B).

Typical use (see GET_LLAMA3.md for the full walkthrough):

    huggingface-cli login            # once, paste your hf_... token
    python download_llama3.py --check-only       # cheap: verify access (few MB)
    python download_llama3.py                     # full download (~16 GB) into HF cache
    python download_llama3.py --local-dir ./models/llama3-8b   # into a folder you choose

Swap the model with --model, e.g.:
    python download_llama3.py --model meta-llama/Llama-3.2-1B-Instruct --check-only
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="Download or verify a gated HF model.")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B",
                    help="HF model id (default: meta-llama/Meta-Llama-3-8B)")
    ap.add_argument("--check-only", action="store_true",
                    help="Only fetch tokenizer+config to confirm access (no big download).")
    ap.add_argument("--local-dir", default=None,
                    help="Download into this folder instead of the shared HF cache.")
    args = ap.parse_args()

    # Token: from `huggingface-cli login` cache, or HF_TOKEN / HUGGING_FACE_HUB_TOKEN env var.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        from huggingface_hub import snapshot_download, whoami
    except ImportError:
        sys.exit("huggingface_hub not installed. Run:  pip install -U 'huggingface_hub[cli]'")

    # Who am I? (confirms a token is available)
    try:
        me = whoami(token=token)
        print(f"Logged in as: {me.get('name', '?')}")
    except Exception:
        print("⚠️  Not logged in. Run `huggingface-cli login` (or set HF_TOKEN). Trying anyway...")

    if args.check_only:
        # Pull only config + tokenizer — a few MB — to prove license + token work.
        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError:
            sys.exit("transformers not installed. Run:  pip install -U transformers")
        try:
            AutoConfig.from_pretrained(args.model, token=token)
            AutoTokenizer.from_pretrained(args.model, token=token)
        except Exception as e:
            print("\n❌ ACCESS FAILED:", e)
            print("   → Make sure the model page shows you have access, and that you've logged in.")
            print("   → See GET_LLAMA3.md (steps 1, 2, 4).")
            sys.exit(1)
        print(f"\n✅ ACCESS OK — your license + token work for {args.model}.")
        print("   Run without --check-only to download the full weights.")
        return

    # Full download.
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") != "1":
        print("tip: `pip install hf_transfer` then `export HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads.")
    print(f"\nDownloading {args.model} (this is large — ~16 GB for 8B models)...")
    try:
        path = snapshot_download(
            repo_id=args.model,
            token=token,
            local_dir=args.local_dir,
            # skip duplicate .pth/.gguf if present; we want the safetensors HF format
            ignore_patterns=["*.pth", "original/*", "*.gguf"],
        )
    except Exception as e:
        print("\n❌ DOWNLOAD FAILED:", e)
        print("   → If this is a gated/401 error, finish steps 1-4 in GET_LLAMA3.md first.")
        sys.exit(1)
    print(f"\n✅ Done. Files are in:\n   {path}")
    print("   transformers will find it automatically via from_pretrained(model_id).")


if __name__ == "__main__":
    main()
