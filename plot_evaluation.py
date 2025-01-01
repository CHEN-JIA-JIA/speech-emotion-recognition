# plot_evaluation.py
# -*- coding: utf-8 -*-
# 将0~7替换为真正的情绪名称输出图形

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model, Wav2Vec2Config
from sklearn.metrics import confusion_matrix, classification_report

############################
# 如果需要谐波过滤，或数据增强:
############################
import torchaudio

def harmonic_filtering(
        waveform,
        sr=16000,
        n_fft=1024,
        hop_length=256,
        threshold_scale=0.6
):
    """
    谐波过滤：对超过阈值幅度的频率成分保留，其他抑制
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    stft_complex = torch.stft(
        waveform.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        return_complex=True
    )
    magnitude = torch.abs(stft_complex)
    mean_val = magnitude.mean()
    threshold = mean_val * threshold_scale

    mask = (magnitude > threshold)
    filtered_stft = stft_complex * mask

    filtered_waveform = torch.istft(
        filtered_stft,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        return_complex=False
    )
    return filtered_waveform.unsqueeze(0)

def augment_waveform(waveform, sr=16000):
    """
    随机加噪 + 时移
    waveform shape: [time]
    """
    wf = waveform.clone().numpy()

    # 1) 随机时移
    if random.random() < 0.5:
        shift_amount = int(sr * 0.2 * random.random())  # 最多 0.2s
        if random.random() < 0.5:
            wf = np.roll(wf, shift_amount)
        else:
            wf = np.roll(wf, -shift_amount)

    # 2) 随机加噪
    if random.random() < 0.5:
        noise_amp = 0.02 * random.random()
        noise = noise_amp * np.random.randn(len(wf))
        wf = wf + noise

    return torch.from_numpy(wf.astype(np.float32))

############################
# 1. RAVDESSDataset
############################
class RAVDESSDataset(Dataset):
    """
    RAVDESS数据集类，支持:
      - 对音频做可选的谐波过滤 (harmonic_filtering)
      - 按固定chunk_size(秒)和chunk_overlap(秒)进行滑窗切分
    """
    def __init__(
            self,
            root_dir,
            sr=16000,
            do_harmonic_filter=False,
            chunk_size=2.0,
            chunk_overlap=0.5
    ):
        self.file_paths = []
        self.labels = []
        self.sr = sr
        self.do_harmonic_filter = do_harmonic_filter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 遍历目录，收集所有 .wav 文件
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".wav"):
                    full_path = os.path.join(root, file)
                    self.file_paths.append(full_path)
                    # 文件名形如: 02-01-03-xx => 第三段为 "03" => int-1 => [0..7]
                    emotion_id_str = file.split("-")[2]
                    emotion_id = int(emotion_id_str) - 1
                    self.labels.append(emotion_id)

    def __len__(self):
        if self.chunk_size is None:
            return len(self.file_paths)
        else:
            total_chunks = 0
            for file_path in self.file_paths:
                info = torchaudio.info(file_path)
                num_frames = info.num_frames
                duration_sec = num_frames / info.sample_rate
                chunk_len = self.chunk_size
                step = chunk_len - self.chunk_overlap
                if step <= 0:
                    raise ValueError("chunk_overlap 太大，导致 step <= 0。")

                chunks_for_this_file = int((duration_sec - self.chunk_overlap) // step)
                if chunks_for_this_file < 1:
                    chunks_for_this_file = 1
                total_chunks += chunks_for_this_file
            return total_chunks

    def __getitem__(self, idx):
        if self.chunk_size is None:
            audio_path = self.file_paths[idx]
            label = self.labels[idx]
            waveform = self.load_and_preprocess(audio_path)
            return waveform, label
        else:
            file_index, chunk_index = self._locate_chunk(idx)
            audio_path = self.file_paths[file_index]
            label = self.labels[file_index]
            waveform = self.load_and_preprocess(audio_path, chunk_index)
            return waveform, label

    def load_and_preprocess(self, audio_path, chunk_index=None):
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.do_harmonic_filter:
            waveform = harmonic_filtering(waveform, sr=self.sr)

        # 滑窗切分
        if self.chunk_size is not None and chunk_index is not None:
            chunk_len_samples = int(self.chunk_size * self.sr)
            overlap_samples = int(self.chunk_overlap * self.sr)
            step = chunk_len_samples - overlap_samples

            total_samples = waveform.shape[-1]
            start = chunk_index * step
            end = start + chunk_len_samples
            if end > total_samples:
                end = total_samples
            waveform = waveform[:, start:end]

        waveform = waveform.to(torch.float32)
        waveform = waveform.squeeze(0)
        # 数据增强
        if random.random() < 0.7:
            waveform = augment_waveform(waveform, sr=self.sr)
        waveform = waveform.to(torch.float32)
        return waveform

    def _locate_chunk(self, global_chunk_idx):
        chunk_len_samples = int(self.chunk_size * self.sr)
        overlap_samples = int(self.chunk_overlap * self.sr)
        step = chunk_len_samples - overlap_samples

        accumulated = 0
        for f_idx, file_path in enumerate(self.file_paths):
            info = torchaudio.info(file_path)
            duration_sec = info.num_frames / info.sample_rate
            chunk_count = int((duration_sec - self.chunk_overlap) // (self.chunk_size - self.chunk_overlap))
            if chunk_count < 1:
                chunk_count = 1
            if global_chunk_idx < (accumulated + chunk_count):
                local_idx = global_chunk_idx - accumulated
                return f_idx, local_idx
            accumulated += chunk_count
        return len(self.file_paths) - 1, 0


############################
# 2. 下游模型
############################
import torch.nn as nn
import torch.nn.functional as F

class LocalAttention(nn.Module):
    """
    局部注意力，用于在 LSTM输出序列的每个时间步
    聚焦其前后一定窗口内的隐状态
    """
    def __init__(self, hidden_dim, attn_window=5):
        super(LocalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_window = attn_window
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_output):
        bsz, seq_len, dim = rnn_output.shape
        outputs = []
        for t in range(seq_len):
            start = max(0, t - self.attn_window)
            end = min(seq_len, t + self.attn_window + 1)
            local_context = rnn_output[:, start:end, :]
            scores = self.attn_score(local_context)
            alpha = F.softmax(scores, dim=1)
            weighted_sum = (local_context * alpha).sum(dim=1)
            outputs.append(weighted_sum.unsqueeze(1))
        return torch.cat(outputs, dim=1)

class EmotionModel(nn.Module):
    def __init__(self, num_emotions=8, hidden_size=128, attn_window=5, feature_dim=768):
        super(EmotionModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.local_attn = LocalAttention(hidden_size * 2, attn_window=attn_window)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_emotions)
        )

    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        attn_out = self.local_attn(lstm_out)
        pooled = attn_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


############################
# 3. 加载Wav2Vec2
############################
from transformers import Wav2Vec2Model, Wav2Vec2Config

def load_wav2vec2_model(model_name="facebook/wav2vec2-base-960h", freeze=True, freeze_layers=4):
    config = Wav2Vec2Config.from_pretrained(model_name)
    wav2vec_model = Wav2Vec2Model.from_pretrained(model_name, config=config)
    if freeze:
        for idx, layer_module in enumerate(wav2vec_model.encoder.layers):
            if idx < freeze_layers:
                for p in layer_module.parameters():
                    p.requires_grad = False
    return wav2vec_model


############################
# 4. collate_fn
############################
from torch.nn.utils.rnn import pad_sequence

def collate_fn_pad(batch):
    waveforms = []
    labels = []
    for (wf, lab) in batch:
        waveforms.append(wf)
        labels.append(lab)
    padded_wf = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return padded_wf, labels_t


############################
# 5. evaluate_model
############################
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(
        dataset,
        checkpoint_path,
        wav2vec_freeze=True,
        wav2vec_freeze_layers=4,
        device=None
):
    """
    加载 checkpoint, 遍历 dataset 得到预测结果
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    wav2vec_model = load_wav2vec2_model(
        freeze=wav2vec_freeze,
        freeze_layers=wav2vec_freeze_layers
    )
    wav2vec_model.load_state_dict(checkpoint["wav2vec_model_state"], strict=False)
    wav2vec_model.to(device)
    wav2vec_model.eval()

    emotion_model = EmotionModel(num_emotions=8, hidden_size=128, feature_dim=768)
    emotion_model.load_state_dict(checkpoint["emotion_model_state"])
    emotion_model.to(device)
    emotion_model.eval()

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_pad
    )

    y_true, y_pred = [], []

    with torch.no_grad():
        for waveforms, labels in data_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            out = wav2vec_model(waveforms)
            hidden_states = out.last_hidden_state
            logits = emotion_model(hidden_states)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cls_report_str = classification_report(y_true, y_pred, digits=4, output_dict=False)
    print("Classification Report:\n", cls_report_str)

    cls_report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)

    return y_true, y_pred, cls_report_dict


############################
# 6. plot_confusion_matrix_and_barchart
############################
def plot_confusion_matrix_and_barchart(
        y_true,
        y_pred,
        cls_report_dict,
        out_dir="evaluation_outputs"
):
    """
    生成并保存:
      - confusion_matrix.png
      - bar_plot_f1.png
    并把数字 0..7 的标签改为相应的文本,如: neutral, calm, happy, sad...
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 指定每个数字对应的情绪标签
    emotion_names = [
        "neutral",    # 0
        "calm",       # 1
        "happy",      # 2
        "sad",        # 3
        "angry",      # 4
        "fearful",    # 5
        "disgust",    # 6
        "surprised"   # 7
    ]

    os.makedirs(out_dir, exist_ok=True)

    # 1) 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=emotion_names,
        yticklabels=emotion_names  # 将坐标轴替换为情绪标签
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    print(f"混淆矩阵已保存到: {cm_path}")

    # 2) 绘制每类的 Precision, Recall, F1
    # cls_report_dict 类似:
    # {
    #   "0": {"precision":..., "recall":..., "f1-score":..., "support":...},
    #   "1": {...}, ...,
    #   "accuracy": ...,
    #   "macro avg": {...}, "weighted avg": {...}
    # }
    # 要把 0~7 => emotion_names
    class_ids = sorted([c for c in cls_report_dict.keys() if c.isdigit()])
    # 转成真实情绪名称
    class_labels = [emotion_names[int(cid)] for cid in class_ids]

    f1_values = [cls_report_dict[c]["f1-score"] for c in class_ids]
    recall_values = [cls_report_dict[c]["recall"] for c in class_ids]
    precision_values = [cls_report_dict[c]["precision"] for c in class_ids]

    x_pos = np.arange(len(class_ids))
    width = 0.2

    plt.figure(figsize=(8, 5))
    plt.bar(x_pos - width, precision_values, width=width, label="Precision")
    plt.bar(x_pos, recall_values, width=width, label="Recall")
    plt.bar(x_pos + width, f1_values, width=width, label="F1-score")
    # 使用 class_labels 替换 x刻度
    plt.xticks(x_pos, class_labels, rotation=45)
    plt.ylim([0, 1])
    plt.xlabel("Emotion Class")
    plt.ylabel("Score")
    plt.title("Per-Class Metrics")
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(out_dir, "bar_plot_f1.png")
    plt.savefig(bar_path)
    plt.show()
    print(f"各类指标条形图已保存到: {bar_path}")


############################
# 7. 如果你有训练时每个 epoch 的日志，则可画 Loss/Acc 曲线
############################
def plot_training_curves(log_path, out_dir="evaluation_outputs"):
    import matplotlib.pyplot as plt

    if not os.path.exists(log_path):
        print(f"日志文件 {log_path} 不存在，无法绘制训练曲线。")
        return

    os.makedirs(out_dir, exist_ok=True)

    with open(log_path, 'r', encoding='utf-8') as f:
        logs = json.load(f)

    epochs = [d["epoch"] for d in logs]
    losses = [d["loss"] for d in logs]
    accuracies = [d["acc"] for d in logs]

    # 绘制 Loss 曲线
    plt.figure(figsize=(8,5))
    plt.plot(epochs, losses, marker='o', label="Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(loss_path)
    plt.show()
    print(f"Loss 曲线已保存到: {loss_path}")

    # 绘制 Accuracy 曲线
    plt.figure(figsize=(8,5))
    plt.plot(epochs, accuracies, marker='o', color='green', label="Accuracy")
    plt.title("Training Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    acc_path = os.path.join(out_dir, "accuracy_curve.png")
    plt.savefig(acc_path)
    plt.show()
    print(f"Accuracy 曲线已保存到: {acc_path}")


############################
# 8. main 示例
############################
def main():
    # 1) 设置数据集路径
    root_dir = r"E:\Speech emotion recognition\Radvess"
    dataset = RAVDESSDataset(
        root_dir=root_dir,
        sr=16000,
        do_harmonic_filter=False, # 是否开启谐波过滤
        chunk_size=2.0,
        chunk_overlap=0.5
    )

    # 2) 评估用的checkpoint(如 best1.pth)
    checkpoint_path = r"E:\Speech emotion recognition\work_dir\best1.pth"
    device = torch.device("cpu")  # 若无GPU则使用CPU

    # 调用 evaluate_model 获取预测值
    y_true, y_pred, cls_report_dict = evaluate_model(
        dataset=dataset,
        checkpoint_path=checkpoint_path,
        wav2vec_freeze=True,
        wav2vec_freeze_layers=4,
        device=device
    )

    # 3) 绘制 混淆矩阵 & 每类指标图（替换 0~7 为真正情绪名称）
    plot_confusion_matrix_and_barchart(
        y_true, y_pred, cls_report_dict,
        out_dir="evaluation_outputs"  # 输出图保存的位置
    )

    # 4) 如果你有每个epoch的日志 -> 画Loss/Acc曲线
    log_file = r"E:\Speech emotion recognition\work_dir\training_epoch_log.json"
    plot_training_curves(log_file, out_dir="evaluation_outputs")

    print("所有图表已生成并保存在 evaluation_outputs 文件夹下。")


if __name__ == "__main__":
    main()
