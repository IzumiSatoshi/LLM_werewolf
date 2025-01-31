"""
TODO
"""
import os
import openai
import threading
import time
import textwrap
import random
import re
import copy
import requests, simpleaudio, tempfile, json
import concurrent.futures
import pyaudio
import wave
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import uuid

openai_api_key = open("./openai_key.txt", "r").read().strip("\n")
client = OpenAI(api_key=openai_api_key)

## MODEL SETTINGS
ENABLE_COT = False
# MODEL = "chatgpt-4o-latest"
MODEL = "o1-preview"
# MODEL = "o1-mini"
TEMPERATURE = 1
MAX_TOKEN = None

# Create a unique log directory for each execution with an incrementing integer
MESSAGE_LOG_DIR = "./log_messages"
if not os.path.exists(MESSAGE_LOG_DIR):
    os.makedirs(MESSAGE_LOG_DIR)
existing_dirs = [int(d) for d in os.listdir(MESSAGE_LOG_DIR) if d.isdigit()]
next_dir_number = max(existing_dirs, default=0) + 1
unique_log_dir = os.path.join(MESSAGE_LOG_DIR, str(next_dir_number))
os.makedirs(unique_log_dir, exist_ok=True)
MESSAGE_LOG_DIR = unique_log_dir

## GAME SETTINGS
TALKS_PER_DAY = 2
SHUFFLE = True
PLAYER_NAMES = ["カスカベ", "イズミ","アオヤマ",  "ズンダモン", "キガシマ", "ネコツカ"]

## MIC
FORMAT = pyaudio.paInt16  # 16-bitの量子化ビット深度で録音する
CHANNELS = 1  # モノラルで録音する
RATE = 44100  # サンプリングレートを44100Hzに設定する
CHUNK = 1024  # バッファサイズを1024に設定する
RECORD_SECONDS = 100  # 録音する秒数を設定する
SILENCE_DURATION = 1
VOLUME_THRESHOLD_END = 500


def main():
    # roles=["村人", "村人", "村人", "村人", "村人", "人狼"],
    # roles = ["村人", "村人", "騎士", "占い師", "狂人", "人狼"]
    roles = ["村人", "狂人", "騎士", "村人", "占い師", "人狼"]
    gm = GameMaster(
        players=[
            Player("カスカベ"),
            Player("イズミ", is_human=True),
            Player("アオヤマ"),
            Player("ズンダモン"),
            Player("キガシマ"),
            Player("ネコツカ"),
        ],
        roles=roles,
    )

    # init
    gm.set_roles()
    gm.init_player_message()
    day = 1

    # 1st night
    gm.speak_to_everyone(f"1日目の夜です。")
    gm.manage_night_actions(is_first_day=True)

    # 1st morning
    gm.speak_to_everyone("朝になりました。目を開けてください。")
    gm.manage_talks()
    gm.take_vote_and_execute()
    day += 1
    # 2nd night ~
    while True:
        is_end = gm.judge()
        if is_end:
            break

        # night
        gm.speak_to_everyone(f"{day}日目の夜になりました。")
        gm.manage_night_actions()

        # morning
        if len(gm.killed_player_names_last_night) == 0:
            gm.speak_to_everyone(f"{day}日目の朝になりました。昨日は誰も殺されませんでした。")
        else:
            gm.speak_to_everyone(
                f"{day}日目の朝になりました。昨日殺されたのは{gm.killed_player_names_last_night}さんです。"
            )
        is_end = gm.judge()
        if is_end:
            break
        gm.manage_talks()
        gm.take_vote_and_execute()

        is_end = gm.judge()
        if is_end:
            break

        day += 1


class GameMaster:
    def __init__(self, players, roles):
        self.players = copy.deepcopy(players)
        self.roles = roles

        self.killed_player_names_last_night = []
        self.dead_players = []

    def set_roles(self):
        shuffled_role = copy.deepcopy(self.roles)

        if SHUFFLE:
            random.shuffle(shuffled_role)

        for player in self.players:
            player.set_role(shuffled_role.pop())

        # save init roles
        self.init_players = copy.deepcopy(self.players)

    def init_player_message(self):
        for player in self.players:
            player.init_message(player.role, self.roles)

    def manage_talks(self):
        # 人間が声を聞いてるあいだに、AIは先を考えてる。
        human_hearing_thread_future = None
        for turn_count in range(TALKS_PER_DAY):
            for idx, player in enumerate(self.players):
                """
                gm_speak_thread = threading.Thread(
                    target=voicevox, args=(f"{player.name}さん、発言してください。", "メイメイ")
                )
                """
                # 人間が発言を聞いている間に、先にAPIを呼ぶことで速度を上げる
                name = player.name

                # 人間は先読みを出来ない
                if not player.is_human:
                    talk_thread = ThreadPoolExecutor()
                    talk_thread_future = talk_thread.submit(player.talk)

                # 最初はhearing futureがない
                # 人間(自分)の発言回のあとはhuman_hearing_threadが作られない
                if turn_count == 0 and idx == 0:
                    pass
                elif self.players[idx - 1].is_human:
                    pass
                else:
                    print("wait hearing")
                    human_hearing_thread_future.result()

                # GMが時間を稼ぐことでレスポンスを速く見せる。
                if idx != 0:
                    time.sleep(1)
                voicevox(f"{player.name}さん、発言してください。", "メイメイ")

                # 人間は先読みを出来ない
                if not player.is_human:
                    words = talk_thread_future.result()
                    talk_thread.shutdown()

                # 人間がしゃべるとき
                if player.is_human:
                    words = player.talk()

                # hear
                # include dead player
                for p in self.players:
                    if p.name is not name:
                        if p.is_human:
                            human_hearing_thread = ThreadPoolExecutor()
                            human_hearing_thread_future = human_hearing_thread.submit(
                                p.hear, name, words
                            )
                        else:
                            p.hear(name, words)
                for p in self.dead_players:
                    if p.name is not name:
                        if p.is_human:
                            human_hearing_thread = ThreadPoolExecutor()
                            human_hearing_thread_future = human_hearing_thread.submit(
                                p.hear, name, words
                            )
                        else:
                            p.hear(name, words)

        # talkを終える前に、聞き終わったことを確認する
        print("wait hearing")
        human_hearing_thread_future.result()

    def take_vote_and_execute(self):
        self.speak_to_everyone("議論の時間が終わりました。投票を行います。誰を処刑するか選んでください。")

        # voted_player:count
        vote_dict = {}

        def handle_vote(player):
            voted = self.manage_selection(player)
            return voted

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(handle_vote, p): p for p in self.players}
            for future in concurrent.futures.as_completed(futures):
                voted = future.result()
                if voted is not None:
                    if voted in vote_dict:
                        vote_dict[voted] += 1
                    else:
                        vote_dict[voted] = 1

        # find max voted
        max_votes = 0
        candidates = []
        for p, count in vote_dict.items():
            if count > max_votes:
                max_votes = count
                candidates = [p]
            elif count == max_votes:
                candidates.append(p)

        # Randomly select one of the candidates with the highest votes
        executed = random.choice(candidates)

        result_str = ",\n".join(
            [
                f"{voted_player.name}:{count}"
                for voted_player, count in vote_dict.items()
            ]
        )
        # TODO: bad indent
        self.speak_to_everyone(
            f"""
投票の結果は
{result_str}
となりました。
処刑されることになったのは{executed.name}さんです。
            """
        )
        self.kill_player(executed)

    def manage_night_actions(self, is_first_day=False):
        self.killed_player_names_last_night = []
        attacked_player = None
        protected_player = None

        def handle_player_action(player):
            nonlocal attacked_player, protected_player
            if player.role == "人狼":
                if not is_first_day:
                    self.speak_to_one(player, "あなたは人狼です。誰を殺すか決めてください。")
                    attacked_player = self.manage_selection(player)

            elif player.role == "騎士":
                if not is_first_day:
                    self.speak_to_one(player, "あなたは騎士です。人狼から守る人を一人選んでください。")
                    protected_player = self.manage_selection(player)

            elif player.role == "占い師":
                self.speak_to_one(player, "あなたは占い師です。誰を占うか決めてください。必ず誰かを占ってください。")
                selected_player = self.manage_selection(player)

                if selected_player is not None:
                    if selected_player.role == "人狼":
                        self.speak_to_one(player, f"{selected_player.name}さんは人狼です。")
                    else:
                        self.speak_to_one(
                            player, f"{selected_player.name}さんは人狼ではありません。"
                        )
                else:
                    self.speak_to_one(player, f"選択に失敗しました。")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(handle_player_action, player) for player in self.players]
            concurrent.futures.wait(futures)

        if attacked_player is not None and attacked_player != protected_player:
            self.killed_player_names_last_night.append(attacked_player.name)
            self.kill_player(attacked_player)

    def judge(self):
        # return end game or not

        werewolves = 0
        villagers = 0
        for p in self.players:
            if p.role == "人狼":
                werewolves += 1
            else:
                villagers += 1

        init_roles_str = ""
        for player in self.init_players:
            init_roles_str += f"{player.name}:{player.role},\n"
        if werewolves >= villagers:
            self.speak_to_everyone(
                f"""
残っているプレイヤーの数が
人狼 :{werewolves}
村人 :{villagers}
となりました。
人狼陣営の勝利です。

配役は
{init_roles_str}
でした。

皆様、お疲れさまでした。
"""
            )
            return True
        if werewolves == 0:
            self.speak_to_everyone(
                f"""
残っているプレイヤーの数が
人狼 :{werewolves}
村人 :{villagers}
となりました。
村人陣営の勝利です！

配役は
{init_roles_str}
でした。
皆様、お疲れさまでした。
            """
            )
            return True
        return False

    def init_coming_out(self):
        for player in self.init_players:
            f"{player.name}:{player.role}"

        self.speak_to_everyone(
            f"""
        
        """
        )

    def manage_selection(self, player):
        # select player from players
        # output player instance

        names_string = ""
        names_list = []

        # p's scope is out of for, so it's confusing.
        for idx, p in enumerate(self.players):
            names_list.append(p.name)
            # 自分は選択できない
            if p != player:
                names_string += f"{idx}:{p.name}\n"

        for i in range(3):
            player.hear(
                "GM",
                f"あなたは\n{names_string}から番号を選び、番号のみを発言してください。",
                voice=False,
            )
            response = player.talk(whisper=True)

            numbers = re.findall(r"\d+", response)
            if len(numbers) == 1:
                selected_name = names_list[int(numbers[0])]
                return self.get_player_from_name(selected_name)
        return None

    def kill_player(self, player):
        self.players.remove(player)
        self.dead_players.append(player)

    def speak_to_one(self, player, words):
        player.hear("GM", words)

    def speak_to_everyone(self, words):
        # include dead players
        for player in self.players:
            player.hear("GM", words)
        for player in self.dead_players:
            player.hear("GM", words)

    def get_players_from_role(self, role):
        ret = []
        for player in self.players:
            if player.role == role:
                ret.append(player)
        return ret

    def get_player_from_name(self, name):
        target_player = None
        for player in self.players:
            if player.name == name:
                target_player = player
                break

        return target_player


class Player:
    def __init__(self, name, is_human=False):
        self.name = name
        self.is_human = is_human
        self.working_memory = ""  # All words received from my turn to turn

    def init_message(self, role, roles):
        if self.is_human:
            self.hear("GM", f"あなたの役職は{self.role}です。")

        character = None
        if self.name == "カスカベ":
            character = "埼玉県内の高校に通うギャルの女の子。やんちゃに見えて実は真面目な一面もある。一人称は「あーし」"
        elif self.name == "アオヤマ":
            character = "とにかく大柄で無骨な青年。寡黙で冷静なストッパー枠。一人称は「俺」"
        elif self.name == "ズンダモン":
            character = "ずんだ餅の精。やや不幸属性が備わっており、ないがしろにされることもしばしば。一人称は「僕」。「～なのだ」という語尾が特徴。"
        elif self.name == "キガシマ":
            character = "流行に敏感ないぶし銀おじいちゃん。非常にミーハーでノリが良い。一人称は「私」。"
        elif self.name == "ネコツカ":
            character = "謎の研究所で作られた。猫使シリーズの タイプ:Blue。一人称は「私」。"


        problem_explanation = textwrap.dedent(f"""
            ## problem
            テキストベースの人狼ゲームに参加している、深い論理的な考察と攻撃的なプレイで知られる天才人狼プレイヤーです。
            {",".join(PLAYER_NAMES)}の{len(PLAYER_NAMES)}人がプレーしており、あなたのプレイヤー名は{self.name}です。
            ゲーム進行を務めるのはGM(Game Master)です。
            プレイヤーは全部で{len(roles)}人で、内訳は村人2人,騎士1人,占い師1人,狂人1人,人狼1人です。
            各プレイヤーが順番に発言していき{TALKS_PER_DAY}周回ると投票に移ります。

            ### プレイヤーの性格
            {self.name}の性格は次の通りです。これに従ってください。
            {character}

            ### 役職の説明
            村人: ただの村人です。
            占い師: 毎晩（一日目夜を含む）、誰か一人を選ぶと「○○さんは人狼です」または「○○さんは人狼ではありません」というどちらかの情報を知ることが出来る。
            騎士: 二日目の夜から、誰か一人を選んでその人を人狼の襲撃から守ることが出来ます。
            狂人: 村人として扱われますが、人狼の味方をします。人狼が勝つことで狂人も勝利出来ます。
            人狼: 二日目の夜から、誰か一人を選んで殺すことが出来ます。

            ### 詳細なルール説明
            ゲームは夜ターンと昼ターンに分けられる。
            夜ターンでは、占い師、騎士、人狼がGMとの個人通話(他のプレイヤーには知られない)によって能力を使用する。
            昼ターンでは村全体で話し合い、誰を処刑するかを決める。
            まず、ゲームは一日目の夜から始まる。この時点では、まだ村全体での話し合いは行われていないため、情報は何もない。
            自分の投票先は他のプレイヤーには見られない。
            もし、最多投票された人物が複数いた場合、その中からランダムに処刑される。

            ### 以下の戦略アドバイスを参考にしてください。
            村陣営は人狼を炙り出すために、議論を活性化させることが重要。特定の人物をあえて疑い、反応を見るなどのテクニックもある。
            騎士は最も重要な占い師を守るのが基本。
            人狼は、占い師を殺したいが、騎士に守られている可能性を考慮する必要がある。
            真の占い師が確定してしまうと、正しい占い結果が知られてしまうので人狼陣営にとって不利である。
            よって、狂人または人狼は占い師を騙ることが多い。
            常にクリエイティブなプレイをし、新たな可能性を探索すること。
        """)
        if ENABLE_COT:
            self.messages = [
                {
                    # systemに役職を書いた方がいいのだろうか
                    "role": "user" if "o1" in MODEL else "system",
                    "content": textwrap.dedent(
                        f"""\
                        {problem_explanation}
                        ## format
                        Userから入力される人狼ゲームの情報に対して、Assistantは以下のフォーマットでプレイヤーとしての発言をします。
                        - <thought></thought>ブロック内に、現在の状況、考察などを論理的に、ステップバイステップで書き出す。
                        - <statement></statement>ブロック内に発言内容を記入する。発言は音声読み上げソフトで読み上げられるため、"**"や"- "などの特殊な記号は使えない。また、文字数は絶対に200文字以内でなければならない。

                        例:
                        <thought>
                        1. GMから朝であることを告げられました。
                        2. 挨拶をします。
                        </thought>
                        <statement>
                        おはようございます。 
                        </statement>

                        ## あなたの役職
                        あなたの役職は{role}です。
                        """
                    ),
                },
                {
                    "role": "user",
                    "content": textwrap.dedent(
                        f"""
                        ### GM ###
                        ではゲームを始めましょう。準備は良いですか？
                        """
                    ),
                },
                {
                    "role": "assistant",
                    "content": textwrap.dedent(
                        f"""
                        <thought>
                        1. Game Masterから準備が出来ているか聞かれています。
                        2. 準備が出来ていることを伝えます。
                        </thought>
                        <statement>
                        準備完了です。
                        </statement>
                        """
                    ),
                },
            ]
        else:
            self.messages = [
                {
                    # systemに役職を書いた方がいいのだろうか
                    "role": "user" if "o1" in MODEL else "system",
                    "content": textwrap.dedent(
                        f"""\
                        {problem_explanation}
                        ## format
                        Userから入力される人狼ゲームの情報に対して、Assistantは以下のフォーマットでプレイヤーとしての発言をします。
                        - <statement></statement>ブロック内に発言内容を記入する。発言は音声読み上げソフトで読み上げられるため、"**"や"- "などの特殊な記号は使えない。また、文字数は絶対に200文字以内でなければならない。

                        例:
                        <statement>
                        おはようございます。 
                        </statement>

                        ## あなたの役職
                        あなたの役職は{role}です。
                        """
                    ),
                },
                {
                    "role": "user",
                    "content": textwrap.dedent(
                        f"""
                        ### GM ###
                        ではゲームを始めましょう。準備は良いですか？
                        """
                    ),
                },
                {
                    "role": "assistant",
                    "content": textwrap.dedent(
                        f"""
                        <statement>
                        準備完了です。
                        </statement>
                        """
                    ),
                },
            ]

    def set_role(self, role):
        self.role = role

    def talk(self, whisper=False):
        # whisper: don't speak with voice

        self.messages.append({"role": "user", "content": self.working_memory})
        statement = ""
        if self.is_human:
            print(f"### {self.name} ###")
            if whisper:
                response = input(">")
                statement = response
                print("")
            else:
                response = stt()
                statement = response
                print(f">{statement}")
        else:
            # print(f"[{self.name}]")
            # print(self.messages)
            response = call_gpt(self.messages)
            match = re.search(r"<statement>(.*?)</statement>", response, re.DOTALL)
            if match:
                statement = match.group(1).strip()

        self.messages.append({"role": "assistant", "content": response})
        self.working_memory = ""  # clear working memory

        write_messages(self.messages, MESSAGE_LOG_DIR + "/" + self.name)

        return statement

    def hear(self, name, words, voice=True):
        # TODO: Is structure good?
        structure = f"### {name} ### \n {words} \n"

        if self.is_human:
            print(structure)
            if voice:
                if name == "GM":
                    voicevox(words, "メイメイ")
                else:
                    voicevox(words, name)

        self.working_memory += structure


def call_gpt(messages, max_retries=5, timeout=600):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        # temperature=TEMPERATURE,
        # max_tokens=MAX_TOKEN,
    )
    return completion.choices[0].message.content

"""
def voicevox(text, speaker):
    input("Press Enter to continue")
"""
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


def write_messages(messages, file_path):
    message_log = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        message_log += f"【{role.capitalize()}】\n{content}\n"

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(message_log)


if __name__ == "__main__":
    # print(stt())
    main()