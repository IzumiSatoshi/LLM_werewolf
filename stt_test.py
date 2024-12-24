import os
import openai
import threading
import time
import textwrap
import random
import re
import copy
import requests, simpleaudio, tempfile, json
import pyaudio
import wave
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

openai_api_key = open("./openai_key.txt", "r").read().strip("\n")
client = OpenAI(api_key=openai_api_key)

PLAYER_NAMES = ["イズミ", "ズンダモン", "カスカベ", "アオヤマ", "キガシマ", "ネコツカ"]
## MIC
FORMAT = pyaudio.paInt16  # 16-bitの量子化ビット深度で録音する
CHANNELS = 1  # モノラルで録音する
RATE = 44100  # サンプリングレートを44100Hzに設定する
CHUNK = 1024  # バッファサイズを1024に設定する
RECORD_SECONDS = 100  # 録音する秒数を設定する
SILENCE_DURATION = 1
VOLUME_THRESHOLD_END = 500

def stt():
    # speach to text with whisper

    audio = pyaudio.PyAudio()  # PyAudioオブジェクトを作成する
    # 録音用のストリームを開く
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    print("録音を開始します...")
    frames = []

    # 録音する時間だけバッファから音声データを読み込む
    # 音量が一定よりも低い時間が続いたら終了
    silence_threshold = int(RATE / CHUNK * SILENCE_DURATION)
    silence_chunks = 0
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)
        volume = np.linalg.norm(audio_data) / np.sqrt(len(audio_data))
        print(volume)
        if volume < VOLUME_THRESHOLD_END:
            silence_chunks += 1
        else:
            silence_chunks = 0

        frames.append(data)
        if silence_chunks > silence_threshold:
            break

    print("録音が終了しました。")

    # 録音用のストリームを閉じる
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with tempfile.TemporaryDirectory() as tmp:
        # 録音した音声データをファイルに保存する
        filepath = f"{tmp}/out.wav"
        waveFile = wave.open(filepath, "wb")
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b"".join(frames))
        waveFile.close()

        # 書き起こし
        with open(filepath, "rb") as f:
            prompt = ",".join(PLAYER_NAMES)
            transcript = client.audio.transcriptions.create(
                model="whisper-1", file=f, language="ja", prompt=prompt
            )

    return transcript.text

print(stt())