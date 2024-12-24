import requests
import time 
import simpleaudio
import tempfile
import json

def voicevox(text, speaker, max_retries=5):
    speaker_id_dict = {
        "ズンダモン": 3,
        "カスカベ": 8,
        "アオヤマ": 13,
        "メイメイ": 14,
        "キガシマ": 53,
        "ネコツカ": 58,
    }
    host = "127.0.0.1"
    port = 50021

    params = (
        ("text", text),
        ("speaker", speaker_id_dict[speaker]),
    )

    for attempt in range(max_retries):
        try:
            response1 = requests.post(f"http://{host}:{port}/audio_query", params=params)
            response1.raise_for_status()

            response2 = requests.post(
                f"http://{host}:{port}/synthesis",
                headers={"Content-Type": "application/json"},
                params=params,
                data=json.dumps(response1.json()),
            )
            response2.raise_for_status()

            with tempfile.TemporaryDirectory() as tmp:
                with open(f"{tmp}/audi.wav", "wb") as f:
                    f.write(response2.content)
                    time.sleep(1)
                    wav_obj = simpleaudio.WaveObject.from_wave_file(f"{tmp}/audi.wav")
                    play_obj = wav_obj.play()
                    play_obj.wait_done()
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                print("Reached maximum retries. Aborting.")
            else:
                time.sleep(1)

text = """
おはようございます。

まず、この混乱した状況を整理しましょう。占い師を名乗る方が3人おり、それぞれ以下の主張をしています：

1. **カスカベさん**：キガシマさんが人狼であると主張。
2. **イズミさん**：カスカベさんが人狼であると主張しながら、なぜかキガシマさんを処刑することを提案。
3. **ズンダモンさん**：キガシマさんが人狼ではないと主張。

ここで注目すべきは、**イズミさんの矛盾した行動**です。彼女はカスカベさんが人狼だと言いながら、キガシマさんを処刑しようと提案しています。これは論理的に破綻しており、村を混乱させる意図があると考えられます。

一方、カスカベさんとズンダモンさんはそれぞれ異なる結果を主張していますが、どちらも一貫した行動をとっています。占い結果が食い違っているのは当然で、人狼ゲームではよくあることです。

また、キガシマさん自身も村人であると主張し、ズンダモンさんの占い結果と一致しています。

以上のことから、**最も疑わしいのはイズミさん**です。彼女が人狼または狂人として村を混乱させようとしている可能性が高いと判断します。

**提案します：今日はイズミさんを処刑しましょう。** 彼女を排除することで、村の情報を整理し、明日の議論につなげることができます。

皆さんのご賛同をお願いします。 
"""
i = 0
while True:
    print(i)
    voicevox(text, "ズンダモン")
    i += 1