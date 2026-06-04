import torch, json, sys
ck = torch.load(
    "model/finetune/checkpoints_track1_formal_global_fw15_s43/best_operating.pt",
    map_location="cpu", weights_only=False,
)
args = ck.get("args", {})
print(json.dumps(args, indent=2, ensure_ascii=False, default=str))
print("SAVED_THRESHOLD:", ck.get("threshold"))
