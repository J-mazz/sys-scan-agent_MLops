# trainer_weighted_sft.py
import math
from typing import List
import torch, torch.nn.functional as F
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

ANSWER_TAG = "### Answer:\n"

def build_hf_dataset(records: List[dict]):
    # records: dicts with input_text, target_text, task, label_score, corr_rho, group_id
    return Dataset.from_list(records)

def add_text_and_weight(ds: Dataset, alpha=1.0, beta=1.0, tau=1.0):
    # compute zstats on label_score
    labels = [float(x) for x in ds["label_score"]]
    mean = sum(labels)/max(1,len(labels))
    var  = sum((x-mean)**2 for x in labels)/max(1,len(labels))
    std  = math.sqrt(var + 1e-8)
    def _map(r):
        z = (float(r["label_score"]) - mean)/(std + 1e-8)
        s = alpha*z + beta*float(r["corr_rho"])
        w = math.log1p(math.exp(tau*s))
        r["loss_weight"] = float(w)
        r["text"] = (f"### Task: {r['task']}\n### Input:\n{r['input_text']}\n{ANSWER_TAG}{r['target_text']}")
        return r
    return ds.map(_map)

class WeightedSFT(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        weights = inputs.pop("loss_weight", None)
        labels  = inputs["labels"]
        outputs = model(**{k:v for k,v in inputs.items() if k!="loss_weight"})
        logits  = outputs.logits
        # causal shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_mask = (shift_labels != -100).float()
        per_tok = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100
        ).view(shift_labels.size())
        tok_counts = loss_mask.sum(dim=1).clamp_min(1.0)
        per_ex = (per_tok * loss_mask).sum(dim=1) / tok_counts
        if weights is None:
            loss = per_ex.mean()
        else:
            w = weights.to(per_ex.device).float()
            w = w / (w.mean() + 1e-8)
            loss = (w * per_ex).mean()
        return (loss, outputs) if return_outputs else loss

def train_weighted_sft(train_ds: Dataset, eval_ds: Dataset=None, model_id="mistral-7b-instruct",
                       max_len=4096, lr=5e-5, use_muon=False):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    resp_ids = tok.encode(ANSWER_TAG, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids=resp_ids, tokenizer=tok, padding_free=True
    )
    cfg = SFTConfig(
        output_dir="out",
        max_seq_length=max_len,
        packing=True,
        padding_free=True,
        eval_packing=False,
        learning_rate=lr,
        warmup_ratio=0.02,
        weight_decay=0.01,
        gradient_checkpointing=True,
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=64,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_steps=1000,
    )
    trainer = WeightedSFT(
        model=model, tokenizer=tok, args=cfg,
        train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator,
    )
    if use_muon:
        from torch.optim import AdamW, Muon
        two_d, other = [], []
        for n,p in trainer.model.named_parameters():
            if not p.requires_grad: continue
            if p.ndim >= 2 and all(k not in n for k in ["embed_tokens","lm_head"]):
                two_d.append(p)
            else:
                other.append(p)
        opt_mu   = Muon([{"params": two_d,  "lr": lr, "weight_decay": 0.01,
                          "momentum": 0.95, "nesterov": True, "ns_steps": 5,
                          "adjust_lr_fn": "match_rms_adamw"}])
        opt_adam = AdamW([{"params": other, "lr": lr, "weight_decay": 0.01}])
        class DuoOpt:
            def __init__(self,a,b): self.a,self.b=a,b
            def zero_grad(self): self.a.zero_grad(); self.b.zero_grad()
            def step(self): self.a.step(); self.b.step()
            def state_dict(self): return {"muon": self.a.state_dict(), "adam": self.b.state_dict()}
            def load_state_dict(self, sd): self.a.load_state_dict(sd["muon"]); self.b.load_state_dict(sd["adam"])
        trainer.create_optimizer = lambda *a, **k: None
        trainer.optimizer = DuoOpt(opt_mu, opt_adam)
    trainer.train()
    return trainer
