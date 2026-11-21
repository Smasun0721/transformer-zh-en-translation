import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from tqdm import tqdm

from dm09_transformer import Transformer

# ================= 1. 基础配置与分词器 =================
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def tokenize_zh(text):
    return text.strip().split()


def tokenize_en(text):
    return text.strip().lower().split()


def build_vocab(file_path, tokenizer, max_vocab_size=16000, min_freq=2):
    counter = Counter()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            counter.update(tokenizer(line))
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX, "<bos>": BOS_IDX, "<eos>": EOS_IDX}
    for word, freq in counter.most_common(max_vocab_size - 4):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


# ================= 2. 极速版 Dataset =================
class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab, max_len=100):
        self.data = []
        with open(src_file, "r", encoding="utf-8") as f_src, open(
            tgt_file, "r", encoding="utf-8"
        ) as f_tgt:
            for src_line, tgt_line in zip(f_src, f_tgt):
                src_tokens = tokenize_zh(src_line)[:max_len]
                tgt_tokens = tokenize_en(tgt_line)[:max_len]
                if len(src_tokens) == 0 or len(tgt_tokens) == 0:
                    continue
                src_ids = [src_vocab.get(w, UNK_IDX) for w in src_tokens]
                tgt_ids = (
                    [BOS_IDX]
                    + [tgt_vocab.get(w, UNK_IDX) for w in tgt_tokens]
                    + [EOS_IDX]
                )
                self.data.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            tgt_ids, dtype=torch.long
        )


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=PAD_IDX
    )
    tgt_padded = nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=PAD_IDX
    )
    return src_padded, tgt_padded


# ================= 3. Mask 与 生成函数 =================
def create_padding_mask(seq):
    return (seq != PAD_IDX).bool()


def create_look_ahead_mask(size):
    return torch.tril(torch.ones((size, size))).bool()


def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    # 实际上这里是 Beam Search 的逻辑
    model.eval()
    k = 3  # 束宽
    with torch.no_grad():
        memory = model.encoder(model.transformer_src_embedding(src), mask=src_mask)

        # 修复 1：尺寸应该是 [k, 1]
        ys = torch.ones(k, 1).fill_(start_symbol).type(torch.long).to(device)
        scores = torch.zeros(k, 1).to(device)

        memory = memory.repeat(k, 1, 1)
        src_mask = src_mask.repeat(k, 1, 1)

        for i in range(max_len - 1):
            # 修复 2：去掉了 .unsqueeze(1)
            tgt_mask = (torch.triu(torch.ones(k, i + 1, i + 1),
                                   diagonal=1) == 0).bool().to(device)

            out = model.decoder(
                model.transformer_target_embedding(ys),
                memory,
                source_mask=src_mask,
                target_mask=tgt_mask,
            )

            logits = model.generator(out[:, -1, :])

            # 防复读惩罚
            for j in range(k):
                for token_id in set(ys[j].tolist()):
                    if token_id not in [BOS_IDX, EOS_IDX, PAD_IDX]:
                        logits[j, token_id] /= 1.5

            log_probs = torch.log_softmax(logits, dim=-1)
            total_probs = scores + log_probs
            total_probs = total_probs.view(-1)
            topk_probs, topk_indices = total_probs.topk(k)

            new_ys = []
            for idx in topk_indices:
                seq_idx = idx // logits.size(-1)
                word_idx = idx % logits.size(-1)
                new_ys.append(torch.cat([ys[seq_idx], word_idx.unsqueeze(0)]))

            ys = torch.stack(new_ys)
            scores = topk_probs.unsqueeze(1)

            if (ys[:, -1] == EOS_IDX).all():
                break

    model.train()
    best_seq_idx = scores.argmax()
    return ys[best_seq_idx]

def tensor_to_text(tensor, vocab):
    idx2word = {v: k for k, v in vocab.items()}
    words = []
    for idx in tensor.tolist():
        if idx in [BOS_IDX, EOS_IDX, PAD_IDX]:
            continue
        words.append(idx2word.get(idx, "<unk>"))
    return " ".join(words)


# ================= 4. 核心 Trainer =================
class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        tgt_vocab,
        src_vocab,
        scheduler=None,
        save_dir="checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.tgt_vocab = tgt_vocab
        self.src_vocab = src_vocab
        self.save_dir = save_dir

        self.scaler = torch.amp.GradScaler(
            "cuda", init_scale=2.0**12, enabled=torch.cuda.is_available()
        )  # 降低初始缩放，更稳定
        os.makedirs(self.save_dir, exist_ok=True)

        self.log_file = "training_log.csv"
        self.log_interval = 20
        self.global_step = 0

        self._init_log_file()

    def _init_log_file(self):
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(
                columns=[
                    "Global_Step",
                    "Epoch",
                    "Type",
                    "Loss",
                    "Acc",
                    "PPL",
                    "LR",
                    "GradNorm",
                ]
            )
            df.to_csv(self.log_file, index=False)

    def _log_step(self, epoch, loss, acc, ppl, lr, grad_norm, log_type="Train"):
        # 防止写入 NaN 破坏日志
        if math.isnan(loss) or math.isinf(loss):
            return

        new_data = pd.DataFrame(
            [
                {
                    "Global_Step": self.global_step,
                    "Epoch": epoch,
                    "Type": log_type,
                    "Loss": loss,
                    "Acc": acc,
                    "PPL": ppl,
                    "LR": lr,
                    "GradNorm": grad_norm,
                }
            ]
        )
        new_data.to_csv(self.log_file, mode="a", header=False, index=False)

    def save_checkpoint(self, epoch, val_loss, filename="last_checkpoint.pt"):
        # 如果 val_loss 是 NaN，坚决不保存
        if math.isnan(val_loss):
            print("⚠️ 警告: Val Loss 为 NaN，跳过本次保存！")
            return

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))

    def load_checkpoint(self, filename="last_checkpoint.pt"):
        path = os.path.join(self.save_dir, filename)
        if os.path.exists(path):
            print(f"🔄 发现断点文件 {path}，正在恢复训练...")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and checkpoint.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.global_step = checkpoint.get("global_step", 0)
            return checkpoint["epoch"], checkpoint["val_loss"]
        return 0, float("inf")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_correct, total_tokens = 0, 0, 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")

        for step, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(self.device, non_blocking=True), tgt.to(
                self.device, non_blocking=True
            )
            tgt_input, tgt_label = tgt[:, :-1], tgt[:, 1:]

            src_mask = create_padding_mask(src).unsqueeze(1)
            tgt_padding_mask = create_padding_mask(tgt_input)
            tgt_look_ahead_mask = create_look_ahead_mask(tgt_input.size(1)).to(
                self.device
            )
            tgt_mask = tgt_padding_mask.unsqueeze(1) & tgt_look_ahead_mask.unsqueeze(0)

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                output = self.model(
                    src, tgt_input, source_mask=src_mask, target_mask=tgt_mask
                )
                loss = self.loss_fn(
                    output.reshape(-1, output.shape[-1]), tgt_label.reshape(-1)
                )

            # 🛡️ NaN 守卫：如果 Loss 爆炸，直接跳过此 batch
            if math.isnan(loss.item()):
                print(f"\n⚠️ Step {self.global_step} Loss 变成 NaN，跳过该批次！")
                self.optimizer.zero_grad()
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.global_step += 1

            total_loss += loss.item()
            preds = output.reshape(-1, output.shape[-1]).argmax(dim=-1)
            mask = tgt_label.reshape(-1) != PAD_IDX
            total_correct += ((preds == tgt_label.reshape(-1)) & mask).sum().item()
            total_tokens += mask.sum().item()

            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / (step + 1)
                avg_acc = total_correct / total_tokens if total_tokens > 0 else 0
                self._log_step(
                    epoch,
                    avg_loss,
                    avg_acc,
                    None,
                    self.optimizer.param_groups[0]["lr"],
                    grad_norm.item(),
                    "Train",
                )
                progress_bar.set_postfix(
                    {
                        "Loss": f"{avg_loss:.4f}",
                        "LR": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                    }
                )

            if self.global_step % 500 == 0:
                self.save_checkpoint(epoch, float("inf"))

        # 防止 total_loss 变 NaN 导致后续全毁
        if math.isnan(total_loss):
            total_loss = 0.0

        return total_loss / len(self.train_loader), total_correct / max(1, total_tokens)

    def evaluate(self, epoch):
        self.model.eval()
        total_loss, total_correct, total_tokens = 0, 0, 0

        with torch.no_grad():
            for src, tgt in tqdm(self.val_loader, desc="[Valid]"):
                src, tgt = src.to(self.device), tgt.to(self.device)
                tgt_input, tgt_label = tgt[:, :-1], tgt[:, 1:]

                src_mask = create_padding_mask(src).unsqueeze(1)
                tgt_padding_mask = create_padding_mask(tgt_input)
                tgt_look_ahead_mask = create_look_ahead_mask(tgt_input.size(1)).to(
                    self.device
                )
                tgt_mask = tgt_padding_mask.unsqueeze(
                    1
                ) & tgt_look_ahead_mask.unsqueeze(0)

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    output = self.model(
                        src, tgt_input, source_mask=src_mask, target_mask=tgt_mask
                    )
                    loss = self.loss_fn(
                        output.reshape(-1, output.shape[-1]), tgt_label.reshape(-1)
                    )

                if math.isnan(loss.item()):
                    continue  # 验证集 NaN 也跳过

                total_loss += loss.item()
                preds = output.reshape(-1, output.shape[-1]).argmax(dim=-1)
                mask = tgt_label.reshape(-1) != PAD_IDX
                total_correct += ((preds == tgt_label.reshape(-1)) & mask).sum().item()
                total_tokens += mask.sum().item()

        avg_loss = total_loss / len(self.val_loader)
        acc = total_correct / max(1, total_tokens)
        ppl = math.exp(avg_loss) if avg_loss < 50 else 99999.0  # 防止 PPL 溢出

        self.global_step += 1
        self._log_step(
            epoch, avg_loss, acc, ppl, self.optimizer.param_groups[0]["lr"], 0, "Val"
        )

        print("\n" + "=" * 50)
        print("📝 模型翻译效果抽样展示 (Greedy Decode):")
        self.model.eval()
        for i in range(3):
            src_sample, tgt_sample = self.val_loader.dataset[i]
            src_sample = src_sample.unsqueeze(0).to(self.device)
            src_mask = create_padding_mask(src_sample).unsqueeze(1)
            pred_ids = greedy_decode(
                self.model,
                src_sample,
                src_mask,
                max_len=100,
                start_symbol=BOS_IDX,
                device=self.device,
            )

            print(f"原句 (ZH): {tensor_to_text(src_sample[0], self.src_vocab)}")
            print(f"标签 (EN): {tensor_to_text(tgt_sample, self.tgt_vocab)}")
            print(f"预测 (EN): {tensor_to_text(pred_ids, self.tgt_vocab)}")
            print("-" * 30)
        print("=" * 50 + "\n")

        return avg_loss, acc, ppl


# ================= 5. 主程序 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    src_file, tgt_file = "data/chinese.txt", "data/english.txt"

    print("正在构建词表并预处理数据（请稍候）...")
    src_vocab = build_vocab(src_file, tokenize_zh)
    tgt_vocab = build_vocab(tgt_file, tokenize_en)

    full_dataset = TranslationDataset(
        src_file, tgt_file, src_vocab, tgt_vocab, max_len=100
    )
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    print(f"数据加载完毕！训练集 Batch 数量: {len(train_loader)}")

    model = Transformer(
        src_vocab_size=len(src_vocab),
        target_vocab_size=len(tgt_vocab),
        d_model=256,
        n_head=8,
        n_layer=4,
        d_ff=1024,
        dropout_p=0.2,
    ).to(device)

    # 🚨 修复：学习率调低至 5e-5，对 Transformer 更加平滑和安全
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    total_epochs = 150
    warmup_epochs = 2
    min_lr_ratio = 0.01  # 🌟 最小学习率是初始值的 10% (如果你想降得更低，可以改成 0.01)

    def lr_lambda(epoch):
        # 1. 预热阶段
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        # 2. 退火阶段
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # 🌟 核心：保证学习率不会跌穿 min_lr_ratio
        return max(min_lr_ratio, cosine_decay)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        tgt_vocab,
        src_vocab,
        scheduler=scheduler,
    )

    start_epoch, best_val_loss = trainer.load_checkpoint()

    for epoch in range(start_epoch, total_epochs):
        train_loss, train_acc = trainer.train_epoch(epoch)
        val_loss, val_acc, val_ppl = trainer.evaluate(epoch)

        print(
            f"✅ Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        trainer.save_checkpoint(epoch + 1, val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(epoch + 1, val_loss, filename="best_model.pt")

        if trainer.scheduler is not None:
            trainer.scheduler.step()


if __name__ == "__main__":
    main()
