import torch
import torch.nn as nn
import os
import jieba  # 引入 jieba 分词

from dm09_transformer import Transformer

# ================= 1. 基础配置 =================
PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_zh(text):
    """与训练阶段一致的按空格分词。"""
    return text.strip().split()


def tokenize_zh_jieba(text):
    """智能分词：如果自带空格，就按空格切；否则用 jieba"""
    text = text.strip()
    if " " in text:
        return text.split()
    return list(jieba.cut(text))


def tokenize_en(text):
    return text.strip().lower().split()


def build_vocab(file_path, tokenizer, max_vocab_size=16000, min_freq=2):
    from collections import Counter

    counter = Counter()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            counter.update(tokenizer(line))
    vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX, "<bos>": BOS_IDX, "<eos>": EOS_IDX}
    for word, freq in counter.most_common(max_vocab_size - 4):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


# ================= 2. Mask 与 Beam Search 解码 =================
def create_padding_mask(seq):
    return (seq != PAD_IDX).bool()


def beam_search_decode(model, src, src_mask, max_len, start_symbol, device, k=3):
    """Beam Search 解码器（已修复维度Bug）"""
    model.eval()
    with torch.no_grad():
        memory = model.encoder(model.transformer_src_embedding(src), mask=src_mask)

        # 🔧 修复 1：批大小是 k (3个句子)，序列长度是 1
        ys = torch.ones(k, 1).fill_(start_symbol).type(torch.long).to(device)
        scores = torch.zeros(k, 1).to(device)  # 记录得分，形状 [k, 1]

        # 复制 Encoder 的输出以匹配 k 个并行序列
        memory = memory.repeat(k, 1, 1)
        src_mask = src_mask.repeat(k, 1, 1)

        for i in range(max_len - 1):
            # 🔧 修复 2：去掉 .unsqueeze(1)，生成形状为 [k, i+1, i+1] 的 Mask，多头注意力内部会自动处理
            tgt_mask = (
                (torch.triu(torch.ones(k, i + 1, i + 1), diagonal=1) == 0)
                .bool()
                .to(device)
            )

            out = model.decoder(
                model.transformer_target_embedding(ys),
                memory,
                source_mask=src_mask,
                target_mask=tgt_mask,
            )

            # out 形状 [k, i+1, d_model]，取最后一个词的概率 [k, vocab_size]
            logits = model.generator(out[:, -1, :])

            # 防复读惩罚
            for j in range(k):
                for token_id in set(ys[j].tolist()):
                    if token_id not in [BOS_IDX, EOS_IDX, PAD_IDX]:
                        logits[j, token_id] /= 1.5

            log_probs = torch.log_softmax(logits, dim=-1)
            total_probs = (
                scores + log_probs
            )  # [k, vocab_size] + [k, 1] = [k, vocab_size]

            # 展平找 Top K
            total_probs = total_probs.view(-1)  # [k * vocab_size]
            topk_probs, topk_indices = total_probs.topk(k)

            # 重构 ys
            new_ys = []
            for idx in topk_indices:
                seq_idx = idx // logits.size(-1)  # 属于原来哪一条候选
                word_idx = idx % logits.size(-1)  # 预测的具体词
                # ys[seq_idx] 形状是 [i+1]，word_idx.unsqueeze(0) 形状是 [1]，拼接后是 [i+2]
                new_ys.append(torch.cat([ys[seq_idx], word_idx.unsqueeze(0)]))

            ys = torch.stack(new_ys)  # 变回 [k, i+2]
            scores = topk_probs.unsqueeze(1)  # 变回 [k, 1]

            # 如果所有候选都生成了 EOS，就提前停止
            if (ys[:, -1] == EOS_IDX).all():
                break

    # 返回得分最高的那一条序列
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


# ================= 3. 主程序 =================
def main():
    print("🚀 正在初始化翻译引擎...")

    # 1. 恢复词表
    src_file, tgt_file = "data/chinese.txt", "data/english.txt"
    if not os.path.exists(src_file):
        print("❌ 找不到 data/chinese.txt")
        return

    src_vocab = build_vocab(src_file, tokenize_zh)  # 与训练阶段保持同一分词方式
    tgt_vocab = build_vocab(tgt_file, tokenize_en)

    # 2. 实例化模型
    model = Transformer(
        src_vocab_size=len(src_vocab),
        target_vocab_size=len(tgt_vocab),
        d_model=256,
        n_head=8,
        n_layer=4,
        d_ff=1024,
        dropout_p=0.2,
    ).to(device)

    # 3. 加载权重
    model_path = "checkpoints/best_model.pt"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型权重 {model_path}，请先完成训练！")
        return

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"✅ 成功加载模型！(训练轮次: {checkpoint['epoch']}, 最佳Val Loss: {checkpoint['val_loss']:.4f})"
    )
    print("=" * 50)

    # 4. 🌟 在这里直接修改你要测试的字符串
    test_sentences = [
        "我 爱 你",  # 带空格，完美匹配训练分布
        "我们 必须 不断 提高 自身 素质",  # 直接抽训练集里的词
        "科学技术 是 第一 生产力",
        "这是 一个 非常 复杂 的 问题",  # 强行打上空格，防止 jieba 切出 <unk>
        "今天 天气 很好"  # 纯 jieba 测试
    ]

    print("📝 开始批量翻译测试：\n")
    for text in test_sentences:
        # 使用 jieba 分词
        tokens = tokenize_zh_jieba(text)

        # 转 ID
        src_ids = [src_vocab.get(w, UNK_IDX) for w in tokens]
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = create_padding_mask(src_tensor).unsqueeze(1)

        # 解码
        pred_ids = beam_search_decode(
            model,
            src_tensor,
            src_mask,
            max_len=100,
            start_symbol=BOS_IDX,
            device=device,
            k=3,
        )

        english_text = tensor_to_text(pred_ids, tgt_vocab)

        # 打印结果
        print(f"🇨🇳 原句: {text}")
        print(f"🔤 切词: {tokens}")
        print(f"🌍 翻译: {english_text}")
        print("-" * 40)


if __name__ == "__main__":
    main()
