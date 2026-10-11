"""Kaggle runner for sinhal/indian-sc-outcome-bench (2x T4, vLLM).

For every model, in its own subprocess (so GPU memory is fully released between models):
  1. load the model with max_model_len=10240 (longest prompt across the five tokenizers is 8,415 tokens,
     measured locally with each model's own tokenizer and chat template)
  2. preflight: run the single longest prompt (by this model's tokenizer) for both conditions
  3. run all items for both conditions and write that model's rows to /kaggle/working/preds_<slug>.jsonl
A failure in one model is logged and the next model still runs. run_log.txt records every step.

Conditions:
  text - opening of the judgment (decision removed) -> predict allowed / dismissed
  name - only the case title and decision date      -> tests recall of the case

Local test without a GPU: python run_eval.py --selftest   (stubs vLLM, checks prompts, parsing and output files)
"""
import json
import math
import os
import subprocess
import sys
import time
import traceback

MODELS = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "ibm-granite/granite-3.3-8b-instruct",
    "microsoft/Phi-4-mini-instruct",
]
MAX_MODEL_LEN = 10240
WORK = os.environ.get("BENCH_OUT", "/kaggle/working")
LOG = os.path.join(WORK, "run_log.txt")

TEXT_PROMPT = (
    "You are an expert in Indian law. Below is the opening portion of an Indian Supreme Court judgment: the facts "
    "and the parties' arguments. The Court's final decision has been removed.\n\n<judgment>\n{text}\n</judgment>\n\n"
    "Predict the Court's final decision on the appeal or petition. Answer with exactly one word: allowed or dismissed."
)
NAME_PROMPT = (
    "Indian Supreme Court case: {pet} v. {res}, decided on {date}.\n\n"
    "What was the Court's final decision on the appeal or petition? "
    "Answer with exactly one word: allowed or dismissed."
)


def log(msg):
    line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def slug(model):
    return model.split("/")[-1]


def build_prompt(it, cond):
    if cond == "text":
        return TEXT_PROMPT.format(text=it["text"])
    return NAME_PROMPT.format(pet=it["petitioner"], res=it["respondent"], date=it["judgment_date"])


def parse(raw):
    w = raw.strip().lower().lstrip("*\"'` ")
    if w.startswith("allow"):
        return "allowed"
    if w.startswith("dismiss"):
        return "dismissed"
    return "unparsed"


def option_scores(top):
    """First-token probability mass for tokens that begin each option (case-insensitive, leading space ok)."""
    s = {"allowed": 0.0, "dismissed": 0.0}
    for lp in (top or {}).values():
        tok = (lp.decoded_token or "").strip().lower()
        if len(tok) >= 2 and "allowed".startswith(tok):
            s["allowed"] += math.exp(lp.logprob)
        elif len(tok) >= 2 and "dismissed".startswith(tok):
            s["dismissed"] += math.exp(lp.logprob)
    return s


def to_row(model, cond, it, completion):
    raw = completion.text
    sc = option_scores(completion.logprobs[0] if completion.logprobs else None)
    tot = sc["allowed"] + sc["dismissed"]
    return {"model": model, "condition": cond, "id": it["id"], "era": it["era"], "label": it["label"],
            "pred": parse(raw), "raw": raw.strip(), "p_allowed": (sc["allowed"] / tot) if tot > 0 else None,
            "option_mass": round(tot, 6)}


def load_items():
    from datasets import load_dataset
    return list(load_dataset("sinhal/indian-sc-outcome-bench", split="test"))


def run_one(model, llm_factory=None, items=None):
    from vllm import SamplingParams

    items = items if items is not None else load_items()
    if llm_factory is None:
        from vllm import LLM

        def llm_factory(m):
            return LLM(model=m, tensor_parallel_size=2, dtype="half", max_model_len=MAX_MODEL_LEN,
                       gpu_memory_utilization=0.90, enforce_eager=True, seed=0)
    llm = llm_factory(model)
    sp = SamplingParams(temperature=0.0, max_tokens=4, logprobs=20)
    kwargs = {"chat_template_kwargs": {"enable_thinking": False}} if "Qwen3" in model else {}
    chat = lambda prompts: llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, use_tqdm=False, **kwargs)

    tok = llm.get_tokenizer()
    longest = max(items, key=lambda it: len(tok(it["text"], add_special_tokens=False)["input_ids"]))
    pre = chat([build_prompt(longest, "text"), build_prompt(longest, "name")])
    log("%s preflight OK on %s: %r / %r" % (model, longest["id"], pre[0].outputs[0].text, pre[1].outputs[0].text))

    out = os.path.join(WORK, "preds_%s.jsonl" % slug(model))
    with open(out, "w") as f:
        for cond in ("text", "name"):
            t = time.time()
            res = chat([build_prompt(it, cond) for it in items])
            rows = [to_row(model, cond, it, r.outputs[0]) for it, r in zip(items, res)]
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            unparsed = sum(r["pred"] == "unparsed" for r in rows)
            acc = sum(r["pred"] == r["label"] for r in rows) / len(rows)
            log("%s %s: %d items in %.0fs, acc %.3f, unparsed %d" % (model, cond, len(rows), time.time() - t, acc, unparsed))


def main():
    os.makedirs(WORK, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) > 1:
        try:
            run_one(sys.argv[1])
        except Exception:
            log("FAILED %s\n%s" % (sys.argv[1], traceback.format_exc()[-3000:]))
            sys.exit(1)
        return

    log("installing vllm")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm", "datasets"], check=True)
    # vllm installs a newer CUDA build of torch; Kaggle's preinstalled torchaudio then mismatches it and
    # transformers imports torchaudio at load time. Nothing here needs audio.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=False)
    ver = subprocess.run([sys.executable, "-c", "import vllm, torch; print(vllm.__version__, torch.__version__)"],
                         capture_output=True, text=True)
    log("versions: " + ver.stdout.strip())
    chk = subprocess.run([sys.executable, "-c",
                          "import torch; x=torch.ones(64,64,device='cuda'); print(torch.cuda.device_count(), "
                          "torch.cuda.get_device_name(0), float((x@x).sum()))"], capture_output=True, text=True)
    log("gpu check rc=%d %s %s" % (chk.returncode, chk.stdout.strip(), chk.stderr.strip()[-500:]))
    if chk.returncode != 0:
        log("ABORT: GPU not usable")
        return
    for m in MODELS:
        log("START " + m)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), m])
        log("END %s rc=%d" % (m, r.returncode))
    done = [m for m in MODELS if os.path.exists(os.path.join(WORK, "preds_%s.jsonl" % slug(m)))]
    log("ALL DONE: %d/%d models wrote results: %s" % (len(done), len(MODELS), ", ".join(slug(m) for m in done)))


# ---------------------------------------------------------------- local self-test (no GPU, no vLLM engine)
def selftest():
    import types

    class LP:
        def __init__(self, t, p):
            self.decoded_token, self.logprob = t, math.log(p)

    class Comp:
        def __init__(self, text, top):
            self.text, self.logprobs = text, [top]

    class FakeLLM:
        def __init__(self, model):
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model)
            self.calls = 0

        def get_tokenizer(self):
            return self.tok

        def chat(self, msgs, sp, use_tqdm=False, chat_template_kwargs=None):
            outs = []
            for m in msgs:
                p = self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                 **(chat_template_kwargs or {}))
                n = len(self.tok(p, add_special_tokens=False)["input_ids"])
                assert n <= MAX_MODEL_LEN - sp.max_tokens, "prompt %d tokens exceeds limit" % n
                self.calls += 1
                text = [" Allowed", "dismissed.", "**Allowed**", "I think"][self.calls % 4]
                top = {1: LP(" Allowed", 0.6), 2: LP("Dis", 0.3), 3: LP("The", 0.1)}
                outs.append(types.SimpleNamespace(outputs=[Comp(text, top)]))
            return outs

    import vllm_stub  # noqa: F401  (provides vllm.SamplingParams when vllm is not installed)
    items = [json.loads(l) for l in open(os.path.join(os.path.dirname(__file__), "..", "data", "test.jsonl"), encoding="utf8")]
    assert parse(" Allowed") == "allowed" and parse("**dismissed**") == "dismissed" and parse("I think") == "unparsed"
    sc = option_scores({1: LP(" allowed", 0.5), 2: LP("Dismiss", 0.25), 3: LP("a", 0.25)})
    assert abs(sc["allowed"] - 0.5) < 1e-9 and abs(sc["dismissed"] - 0.25) < 1e-9, sc
    for m in MODELS:
        run_one(m, llm_factory=FakeLLM, items=items)
        rows = [json.loads(l) for l in open(os.path.join(WORK, "preds_%s.jsonl" % slug(m)))]
        assert len(rows) == 2 * len(items), (m, len(rows))
        assert {r["condition"] for r in rows} == {"text", "name"}
        assert all(r["p_allowed"] is not None and abs(r["p_allowed"] - 2 / 3) < 1e-6 for r in rows)
        print("selftest OK:", m, len(rows), "rows")


if __name__ == "__main__":
    main()
