# 系统交互界面
# 目前最好的是：E:\Speech emotion recognition\work_dir\best1.pth

import tkinter as tk
import tkinter.messagebox as msg
import pyaudio
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Config
import sys
import os

#######################
# 1. 情绪标签列表
#######################
EMOTION_LABELS = [
    "neutral",    # index 0
    "calm",       # index 1
    "happy",      # index 2
    "sad",        # index 3
    "angry",      # index 4
    "fearful",    # index 5
    "disgust",    # index 6
    "surprised"   # index 7
]

#######################
# 2. 下游模型定义
#######################
class LocalAttention(torch.nn.Module):
    def __init__(self, hidden_dim, attn_window=5):
        super(LocalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_window = attn_window
        self.attn_score = torch.nn.Linear(hidden_dim, 1)

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

class EmotionModel(torch.nn.Module):
    def __init__(self, num_emotions=8, hidden_size=128, attn_window=5, feature_dim=768):
        super(EmotionModel, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.local_attn = LocalAttention(hidden_size*2, attn_window=attn_window)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size*2, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_emotions)
        )

    def forward(self, features):
        lstm_out, _ = self.lstm(features)
        attn_out = self.local_attn(lstm_out)
        pooled = attn_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

###############################
# 3. 实时推理函数
###############################
def real_time_inference(audio_chunk, wav2vec_model, emotion_model, device=None):
    if device is None:
        device = torch.device("cpu")
    if audio_chunk.dim() == 1:
        audio_chunk = audio_chunk.unsqueeze(0)  # => [1, time]
    audio_chunk = audio_chunk.to(device)

    with torch.no_grad():
        # 1) 用 Wav2Vec2 提取特征
        out = wav2vec_model(audio_chunk)
        hidden_states = out.last_hidden_state

        # 2) 下游模型预测
        logits = emotion_model(hidden_states)
        pred_idx = torch.argmax(logits, dim=1).item()

    return pred_idx

###############################
# 4. PyAudio 录音 + Tkinter GUI
###############################
class SpeechEmotionGUI:
    def __init__(self, master, wav2vec_model, emotion_model, device):
        self.master = master
        self.master.title("Real-Time Speech Emotion Recognition")

        self.wav2vec_model = wav2vec_model
        self.emotion_model = emotion_model
        self.device = device

        self.sample_rate = 16000
        self.channels = 1
        self.record_seconds = 2  # 录音时长(秒)
        self.chunk = 1024
        self.format = pyaudio.paInt16

        # 显示所有可识别的情绪类别
        emotion_list_str = "可识别的情绪类别：\n" + ", ".join(
            f"[{i}]{label}" for i, label in enumerate(EMOTION_LABELS)
        )
        self.emotion_list_label = tk.Label(self.master, text=emotion_list_str, fg="blue")
        self.emotion_list_label.pack(pady=5)

        self.status_label = tk.Label(self.master, text="点击【开始录音】进行录音并识别情感")
        self.status_label.pack(pady=5)

        self.record_button = tk.Button(self.master, text="开始录音", command=self.record_audio)
        self.record_button.pack(pady=10)

    def record_audio(self):
        self.status_label.config(text="录音中...")
        self.master.update()

        # 录音
        audio_data = self.capture_audio()

        # 转成 PyTorch tensor
        waveform = torch.from_numpy(audio_data).float()

        # 推理
        pred_idx = real_time_inference(
            audio_chunk=waveform,
            wav2vec_model=self.wav2vec_model,
            emotion_model=self.emotion_model,
            device=self.device
        )

        pred_emotion = EMOTION_LABELS[pred_idx] if 0 <= pred_idx < len(EMOTION_LABELS) else f"未知({pred_idx})"

        self.status_label.config(text="录音结束")
        msg.showinfo("识别结果", f"预测情感类别: {pred_emotion}")

    def capture_audio(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        frames = []
        num_frames = int(self.sample_rate / self.chunk * self.record_seconds)

        for _ in range(num_frames):
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        pa.terminate()

        audio_bytes = b''.join(frames)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        # 归一化到 -1.0 ~ 1.0
        audio_array = audio_array.astype(np.float32) / 32767.0
        return audio_array

###################################
# 5. main - 加载 best1.pth 并启动GUI
###################################
def main():
    # 1) 载入保存的 "best1.pth"
    best_model_path = r"E:\Speech emotion recognition\work_dir\best1.pth"
    if not os.path.exists(best_model_path):
        print(f"模型文件 {best_model_path} 不存在，请先检查路径。")
        sys.exit(1)

    checkpoint = torch.load(best_model_path, map_location="cpu")
    print(f"Loaded checkpoint from {best_model_path}")

    # 2) 初始化 wav2vec2
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", config=config)

    # 有些 pretrained 模型可能包含额外 key，所以用 strict=False
    wav2vec_model.load_state_dict(checkpoint["wav2vec_model_state"], strict=False)

    # 3) 初始化下游情感模型 (要和你训练时的结构保持一致)
    emotion_model = EmotionModel(
        num_emotions=8,    # 你在训练时的情绪类别数
        hidden_size=128,   # 训练时的 hidden_size
        attn_window=5,     # 训练时的 attn_window
        feature_dim=768    # wav2vec2-base-960h 的 hidden_size 是 768
    )
    emotion_model.load_state_dict(checkpoint["emotion_model_state"])

    # 4) 选择设备(一般演示可用CPU)
    device = torch.device("cpu")
    wav2vec_model.to(device)
    emotion_model.to(device)
    wav2vec_model.eval()
    emotion_model.eval()

    # 5) 启动 Tkinter GUI
    root = tk.Tk()
    app = SpeechEmotionGUI(root, wav2vec_model, emotion_model, device)
    root.mainloop()


if __name__ == "__main__":
    main()
