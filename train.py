# resume_train_model.py
# 最后的


import os
import time
import torch
import torchaudio
import torch.nn as nn
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Model, Wav2Vec2Config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import numpy as np

############################
# 1. 数据增强 (可选)
############################
def augment_waveform(waveform, sr=16000):
    wf = waveform.clone().numpy()
    if random.random() < 0.5:
        shift_amount = int(sr * 0.2 * random.random())
        if random.random() < 0.5:
            wf = np.roll(wf, shift_amount)
        else:
            wf = np.roll(wf, -shift_amount)
    if random.random() < 0.5:
        noise_amp = 0.02 * random.random()
        noise = noise_amp * np.random.randn(len(wf))
        wf = wf + noise
    return torch.from_numpy(wf.astype(np.float32))

############################
# 2. 谐波过滤 (可选)
############################
def harmonic_filtering(
        waveform,
        sr=16000,
        n_fft=1024,
        hop_length=256,
        threshold_scale=0.6
):
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

############################
# 3. RAVDESS数据集 (带滑窗策略)
############################
class RAVDESSDataset(Dataset):
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

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".wav"):
                    full_path = os.path.join(root, file)
                    self.file_paths.append(full_path)

                    # 情感ID第三段 => 减1让范围变[0..7]
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
                    raise ValueError("chunk_overlap too large.")
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
# 4. 下游模型
############################
class LocalAttention(nn.Module):
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
# 5. Wav2Vec2加载
############################
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
# 6. collate_fn
############################
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
# 7. 训练函数 (带 Best/Last 保存 + EarlyStopping + 时间戳)
############################
def train_emotion_model(
        dataset,
        wav2vec_model,
        batch_size=12,
        epochs=80,
        lr=1e-5,
        patience=5,              # Early Stopping 容忍次数 (连续多少轮没提升就停)
        num_emotions=8,
        hidden_size=128,
        device=None,
        optimizer=None,
        scheduler=None,
        emotion_model=None,
        best_path="best.pth",
        last_path="last.pth"
):
    """
    允许传入已经创建好的 optimizer、scheduler、emotion_model；
    如果没有，就在函数内新建。新增功能：
    1) 每个epoch前打印时间戳
    2) 保存 best (F1最高) 和 last (最后一轮) 模型
    3) 提供 Early Stopping，当连续 'patience' 次没有提升F1，则停止训练
    """

    import datetime  # 用于打印更漂亮的时间

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_dim = wav2vec_model.config.hidden_size

    # 如果上层没传 model 进来，就新建一个
    if emotion_model is None:
        emotion_model = EmotionModel(num_emotions=num_emotions, hidden_size=hidden_size, feature_dim=feature_dim)
    emotion_model.to(device)

    # 构建损失函数
    criterion = nn.CrossEntropyLoss()

    # 如果没传 optimizer/scheduler，就在这里创建
    if optimizer is None or scheduler is None:
        param_groups = [
            {
                "params": [p for p in wav2vec_model.parameters() if p.requires_grad],
                "lr": lr * 0.1
            },
            {
                "params": [p for p in emotion_model.parameters() if p.requires_grad],
                "lr": lr
            }
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_pad
    )

    best_f1 = -1.0
    best_epoch = 0
    num_bad_epochs = 0  # 连续没有提升F1的epoch计数

    final_epoch_loss = None
    final_acc = None
    final_prec = None
    final_rec = None
    final_f1 = None

    for epoch in range(epochs):
        # ==== 每个 epoch 前 打印时间戳 ====
        current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n=== Start Epoch [{epoch+1}/{epochs}] at {current_time_str} ===")

        emotion_model.train()
        all_preds = []
        all_labels = []
        running_loss = 0.0
        total_samples = 0

        for waveforms, labels in data_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            out = wav2vec_model(waveforms)
            hidden_states = out.last_hidden_state

            logits = emotion_model(hidden_states)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * waveforms.size(0)
            total_samples += waveforms.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        scheduler.step()

        epoch_loss = running_loss / total_samples
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Epoch [{epoch + 1}/{epochs}] - "
              f"Loss: {epoch_loss:.4f}, "
              f"Acc: {acc:.4f}, "
              f"Precision: {prec:.4f}, "
              f"Recall: {rec:.4f}, "
              f"F1: {f1:.4f}")

        # 更新最终epoch的指标
        final_epoch_loss = epoch_loss
        final_acc = acc
        final_prec = prec
        final_rec = rec
        final_f1 = f1

        # ---- 检查是否当前最佳 (以 F1 为准) ----
        torch.save({
            "epoch": epoch + 1,
            "wav2vec_model_state": wav2vec_model.state_dict(),
            "emotion_model_state": emotion_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metrics": {
                "Loss": epoch_loss,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1
            }
        }, last_path)  # 每轮都保存 last.pth

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            num_bad_epochs = 0  # 重置
            # 保存最佳模型
            torch.save({
                "epoch": epoch + 1,
                "wav2vec_model_state": wav2vec_model.state_dict(),
                "emotion_model_state": emotion_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "metrics": {
                    "Loss": epoch_loss,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1": f1
                }
            }, best_path)
            print(f"  --> New best F1 = {f1:.4f}, model saved to {best_path}")
        else:
            num_bad_epochs += 1
            print(f"  --> F1 did not improve for {num_bad_epochs} epoch(s).")

        # ---- Early Stopping ----
        if num_bad_epochs >= patience:
            print(f"Early stopping: F1 has not improved in the last {patience} epochs.")
            break

    return emotion_model, optimizer, scheduler, (final_epoch_loss, final_acc, final_prec, final_rec, final_f1, best_f1, best_epoch)

############################
# 8. 推理函数
############################
def real_time_inference(
        audio_signal,
        wav2vec_model,
        emotion_model,
        sr=16000,
        chunk_size=2.0,
        chunk_overlap=0.5,
        device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if audio_signal.dim() == 2:
        audio_signal = audio_signal.squeeze(0)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio_signal = resampler(audio_signal.unsqueeze(0)).squeeze(0)
        sr = 16000

    chunk_len_samples = int(chunk_size * sr)
    overlap_samples = int(chunk_overlap * sr)
    step = chunk_len_samples - overlap_samples

    total_samples = audio_signal.shape[0]
    start = 0
    logits_list = []

    with torch.no_grad():
        while True:
            end = start + chunk_len_samples
            if end > total_samples:
                end = total_samples
            chunk = audio_signal[start:end]

            if len(chunk) < chunk_len_samples:
                pad_len = chunk_len_samples - len(chunk)
                chunk = F.pad(chunk, (0, pad_len))

            chunk = chunk.unsqueeze(0).to(device)
            wav_out = wav2vec_model(chunk)
            hidden_states = wav_out.last_hidden_state

            chunk_logits = emotion_model(hidden_states)
            logits_list.append(chunk_logits.cpu())

            start += step
            if start >= total_samples:
                break

    avg_logits = torch.mean(torch.cat(logits_list, dim=0), dim=0, keepdim=True)
    pred = torch.argmax(avg_logits, dim=1).item()
    return pred


############################
# 9. main - 从已有的checkpoint恢复并继续训练
############################
if __name__ == "__main__":
    # 准备数据集
    root_dir = r"E:\Speech emotion recognition\Radvess"
    dataset = RAVDESSDataset(
        root_dir=root_dir,
        sr=16000,
        do_harmonic_filter=False,
        chunk_size=2.0,
        chunk_overlap=0.5
    )
    print("Dataset size:", len(dataset))

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 准备wav2vec2模型 (同之前冻结设置)
    wav2vec_model = load_wav2vec2_model(freeze=True, freeze_layers=4)
    wav2vec_model.to(device)

    # 1) 先创建一个 emotion_model 实例（结构要与之前训练时一致）
    feature_dim = wav2vec_model.config.hidden_size
    resumed_emotion_model = EmotionModel(num_emotions=8, hidden_size=128, feature_dim=feature_dim)
    resumed_emotion_model.to(device)

    # 2) 创建一个空的优化器 & 调度器（以备后面加载）
    param_groups = [
        {
            "params": [p for p in wav2vec_model.parameters() if p.requires_grad],
            "lr": 1e-5 * 0.1
        },
        {
            "params": [p for p in resumed_emotion_model.parameters() if p.requires_grad],
            "lr": 1e-5
        }
    ]
    resumed_optimizer = torch.optim.AdamW(param_groups, lr=1e-5, weight_decay=1e-2)
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=5)

    # 3) 加载之前保存的checkpoint
    checkpoint_path = r"E:\Speech emotion recognition\work_dir\emotion_model05_resumed.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 4) 恢复模型权重
    wav2vec_model.load_state_dict(checkpoint["wav2vec_model_state"])
    resumed_emotion_model.load_state_dict(checkpoint["emotion_model_state"])

    # 5) 恢复优化器和调度器状态
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
    resumed_scheduler.load_state_dict(checkpoint["scheduler_state"])

    print(f"Resumed from checkpoint: {checkpoint_path}")
    print("Previous metrics:", checkpoint["metrics"])

    # 6) 继续训练 (例如，再来 80 个 epoch)，并启用 EarlyStopping, 保存 best/last
    extra_epochs = 5
    best_ckpt_path = r"E:\Speech emotion recognition\work_dir\best1.pth"
    last_ckpt_path = r"E:\Speech emotion recognition\work_dir\last1.pth"
    resumed_emotion_model, resumed_optimizer, resumed_scheduler, final_metrics = train_emotion_model(
        dataset=dataset,
        wav2vec_model=wav2vec_model,
        batch_size=12,
        epochs=extra_epochs,
        lr=1e-5,
        device=device,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        emotion_model=resumed_emotion_model,
        best_path=best_ckpt_path,
        last_path=last_ckpt_path,
        patience=5  # 如果5个epoch都没有提升F1，就会提前停
    )

    # final_metrics 返回: (final_loss, final_acc, final_prec, final_rec, final_f1, best_f1, best_epoch)
    final_loss, final_acc, final_prec, final_rec, final_f1, best_f1, best_epoch = final_metrics

    # 7) 单独保存最终日志 (JSON)，包含最后一轮和最优模型的信息
    new_log_path = r"E:\Speech emotion recognition\work_dir\final_log1.json"
    final_log_data = {
        "Final_Epoch_Loss": final_loss,
        "Final_Accuracy": final_acc,
        "Final_Precision": final_prec,
        "Final_Recall": final_rec,
        "Final_F1": final_f1,
        "Best_F1": best_f1,
        "Best_Epoch": best_epoch
    }
    with open(new_log_path, 'w', encoding='utf-8') as f:
        json.dump(final_log_data, f, indent=4)

    print("Training finished.")
    print(f"Last model saved to {last_ckpt_path}")
    print(f"Best model (F1={best_f1:.4f}) saved to {best_ckpt_path}")
    print(f"Final log saved to {new_log_path}")
