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
ENABLE_COT = True
# MODEL = "chatgpt-4o-latest"
MODEL = "o1"
# MODEL = "o3-mini"
REASONING_EFFORT = "high"
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
PLAYER_NAMES = ["Sophia", "Izumi","Zoe", "Zeal", "Levi", "Vi"]


def main():
    roles = ["villager", "maniac", "hunter", "villager", "seer", "werewolf"]
    gm = GameMaster(
        players=[
            Player("Sofia"),
            Player("Izumi", is_human=True),
            Player("Zoe"),
            Player("Zeal"),
            Player("Levi"),
            Player("Vi"),
        ],
        roles=roles,
    )

    # init
    gm.set_roles()
    gm.init_player_message()
    day = 1

    # 1st night
    gm.speak_to_everyone(f"The First Night")
    gm.manage_night_actions(is_first_day=True)

    # 1st morning
    gm.speak_to_everyone("The first Morning. Good morning everyone!")
    gm.manage_talks()
    gm.take_vote_and_execute()
    day += 1
    # 2nd night ~
    while True:
        is_end = gm.judge()
        if is_end:
            break

        # night
        gm.speak_to_everyone(f"This is {day}th Night")
        gm.manage_night_actions()

        # morning
        if len(gm.killed_player_names_last_night) == 0:
            gm.speak_to_everyone(f"This is {day}th Morning. Noone was killed at night.")
        else:
            gm.speak_to_everyone(
                f"This is {day}th Morning. {gm.killed_player_names_last_night} was killed at night."
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
                # 人間が発言を聞いている間に、先にAPIを呼ぶことで速度を上げる
                name = player.name
                words = player.talk()

                # hear
                # include dead player
                for p in self.players:
                    if p.name is not name:
                        p.hear(name, words)
                for p in self.dead_players:
                        p.hear(name, words)

    def take_vote_and_execute(self):
        self.speak_to_everyone("The discussion is over. Please decide who to lynch.")

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
The resule of votes were following.
{result_str}
{executed.name} was lynched
            """
        )
        self.kill_player(executed)

    def manage_night_actions(self, is_first_day=False):
        self.killed_player_names_last_night = []
        attacked_player = None
        protected_player = None

        def handle_player_action(player):
            nonlocal attacked_player, protected_player
            if player.role == "werewolf":
                if not is_first_day:
                    self.speak_to_one(player, "You are a werewolf, please decide who to kill.")
                    attacked_player = self.manage_selection(player)

            elif player.role == "hunter":
                if not is_first_day:
                    self.speak_to_one(player, "You are a hunter. Please decide who to protect from werewolf.")
                    protected_player = self.manage_selection(player)

            elif player.role == "seer":
                self.speak_to_one(player, "you are a seer. Please choose one person to reveal their identity. You need to choose one person.")
                selected_player = self.manage_selection(player)

                if selected_player is not None:
                    if selected_player.role == "werewolf":
                        self.speak_to_one(player, f"{selected_player.name} is werewolf.")
                    else:
                        self.speak_to_one(
                            player, f"{selected_player.name} is not werewolf."
                        )
                else:
                    self.speak_to_one(player, f"You failed to choose person.")
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
            if p.role == "werewolf":
                werewolves += 1
            else:
                villagers += 1

        init_roles_str = ""
        for player in self.init_players:
            init_roles_str += f"{player.name}:{player.role},\n"
        if werewolves >= villagers:
            self.speak_to_everyone(
                f"""
The werewolf side won!

werewolf :{werewolves}
villager :{villagers}

{init_roles_str}
"""
            )
            return True
        if werewolves == 0:
            self.speak_to_everyone(
                f"""
The villager side won!

werewolf :{werewolves}
villager :{villagers}

{init_roles_str}
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
            self.speak_to_one(player, f"You need to choose one number from following. \n{names_string}")
            
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
        player.hear("GM", words, whisper=True)

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
            self.hear("GM", f"You are {self.role}")

        problem_explanation = textwrap.dedent(f"""
            ## Problem
            You are a genius Werewolf (Mafia) player, known for your deep logical analysis and aggressive playstyle, participating in a text-based Werewolf game.
            There are {len(PLAYER_NAMES)} players in total: {",".join(PLAYER_NAMES)}, and your player name is {self.name}.
            The game is managed by the Game Master (GM).

            The game consists of {len(roles)} players with the following roles:

            Villager ×2: Ordinary villagers.
            Seer ×1: Each night (including Night 1), the Seer can choose one player to reveal whether they are a Werewolf or not. The result is whether 'player_name is werewolf' or 'player_name is not werewolf'
            Hunter ×1: Starting from the second night, the hunter can protect one player from being attacked by the Werewolf.
            Maniac ×1: Counts as a Villager but secretly supports the Werewolves. The Maniac wins if the Werewolves win.
            Werewolf ×1: Starting from the second night, the Werewolf can choose one player to eliminate.
            Each player takes turns speaking, and after {TALKS_PER_DAY} rounds of discussion, the game proceeds to voting.

            ## Detailed Rules
            The game alternates between Night Turns and Day Turns.

            Night Turn:
            The Seer, Hunter, and Werewolf use their abilities through private communication with the GM (hidden from other players).
            Day Turn:
            All players discuss and decide who to eliminate.
            The game starts on Night 1, during which no discussions take place, meaning no public information is available at the beginning.
            Each player's vote is private and not visible to others.
            If multiple players receive the highest number of votes, one of them is chosen randomly for elimination.
            ## Strategy Advice
            Villagers: To expose the Werewolf, create active discussions. A common technique is to deliberately cast doubt on specific players and observe their reactions.
            Hunter: Prioritize protecting the Seer, as they are the most valuable player.
            Werewolf: While eliminating the Seer is crucial, be cautious, as the Hunter might protect them.
            Seer Exposure: Seer should claim their role and provide info at first day. If the true Seer is revealed, their verified information becomes a major threat to the Werewolves. Therefore, the Maniac must fake to be the Seer to create confusion.
            Creative Play: Always think outside the box and explore new strategic possibilities.
            Discussion Language: All discussions must be conducted in English.
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
                        Given the game info from 'User', Assistant will answer by following format.
                        - <thought></thought> block : you need to write step-by-step thought process.
                        - <statement></statement> block : you need to write your statement.

                        Example:
                        <thought>
                        1. GM told me it is morning.
                        2. I will greet to others.
                        </thought>
                        <statement>
                        Good morning
                        </statement>

                        ## Your Role
                        Your role is {role}.
                        """
                    ),
                },
                {
                    "role": "user",
                    "content": textwrap.dedent(
                        f"""
                        ### GM ###
                        Ok, let's start the game.
                        """
                    ),
                },
                {
                    "role": "assistant",
                    "content": textwrap.dedent(
                        f"""
                        <thought>
                        1. GM is asking quetion whether I am ready or not.
                        2. I will say yes.
                        </thought>
                        <statement>
                        yes
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
                        Given the game info from 'User', Assistant will answer by following format.
                        - <statement></statement> block : you need to write your statement.

                        Example:
                        <statement>
                        Good morning
                        </statement>

                        ## Your Role
                        Your role is {role}.
                        """
                    ),
                },
                {
                    "role": "user",
                    "content": textwrap.dedent(
                        f"""
                        ### GM ###
                        Ok, let's start the game.
                        """
                    ),
                },
                {
                    "role": "assistant",
                    "content": textwrap.dedent(
                        f"""
                        <statement>
                        I'm ready.
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
                response = input(">")
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

        write_messages(self.messages, MESSAGE_LOG_DIR + "/" + self.name + ".txt")

        return statement

    def hear(self, name, words, whisper=False, voice=True):
        # TODO: Is structure good?
        if whisper:
            structure = f"### {name} (whispering to you) ### \n {words} \n"
        else:
            structure = f"### {name} ### \n {words} \n"


        if self.is_human:
            print(structure)

        self.working_memory += structure


def call_gpt(messages, max_retries=5, timeout=600):
    if REASONING_EFFORT is None:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            # temperature=TEMPERATURE,
            # max_tokens=MAX_TOKEN,
        )
    else:
        completion = client.chat.completions.create(
            model=MODEL,
            reasoning_effort=REASONING_EFFORT,
            messages=messages,
            # temperature=TEMPERATURE,
            # max_tokens=MAX_TOKEN,
        )

    return completion.choices[0].message.content

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