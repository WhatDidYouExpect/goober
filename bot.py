import discord
from discord.ext import commands, tasks
from discord import app_commands
import json
import nltk
from nltk.tokenize import word_tokenize
import random
import os
import time
import asyncio
import re
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
import requests
import platform
import subprocess
import psutil
import pickle
import hashlib
from better_profanity import profanity
from config import *
import traceback
import shutil
from nltk.sentiment.vader import SentimentIntensityAnalyzer
if platform.system() == 'Darwin':
    try:
        import metal
        device = metal.MTLCreateSystemDefaultDevice()
        if device:
            print(f"{GREEN}Metal is supported on this mac, Attemping to use tensorflow-metal.{RESET}")
            import tensorflow-metal as tfmetal
            tf.config.set_visible_devices([], 'GPU')
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            print(f"{YELLOW}Metal not supported on this mac. Using regular TensorFlow.{RESET}")
    except ImportError:
        print(f"{YELLOW}tensorflow-metal not available. using regular TensorFlow.{RESET}")
else:
    print(f"{DEBUG}Not running on macOS. using regular TensorFlow.{RESET}")

import tensorflow as tf
import numpy as np
from tensorflow import keras
Tokenizer = keras.preprocessing.text.Tokenizer
pad_sequences = keras.preprocessing.sequence.pad_sequences
Sequential = keras.models.Sequential
Embedding, LSTM, Dense, Dropout, Bidirectional = (
    keras.layers.Embedding, keras.layers.LSTM, keras.layers.Dense, keras.layers.Dropout, keras.layers.Bidirectional
)

tf.get_logger().setLevel('ERROR') 

analyzer = SentimentIntensityAnalyzer()

print(splashtext) # you can use https://patorjk.com/software/taag/ for 3d text or just remove this entirely

def download_json():
    locales_dir = "locales"
    response = requests.get(f"{VERSION_URL}/goob/locales/{LOCALE}.json")
    if response.status_code == 200:

        if not os.path.exists(locales_dir):
            os.makedirs(locales_dir)
        file_path = os.path.join(locales_dir, f"{LOCALE}.json")
        if os.path.exists(file_path):
            return
        else:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(response.text)

    if not os.path.exists(os.path.join(locales_dir, "en.json")):
        
        response = requests.get(f"{VERSION_URL}/goob/locales/en.json")
        if response.status_code == 200:
            with open(os.path.join(locales_dir, "en.json"), "w", encoding="utf-8") as file:
                file.write(response.text)

download_json()
def load_translations():
    translations = {}
    translations_dir = os.path.join(os.path.dirname(__file__), "locales")
    
    for filename in os.listdir(translations_dir):
        if filename.endswith(".json"):
            lang_code = filename.replace(".json", "")
            with open(os.path.join(translations_dir, filename), "r", encoding="utf-8") as f:
                translations[lang_code] = json.load(f)
    
    return translations

translations = load_translations()

def get_translation(lang: str, key: str):
    lang_translations = translations.get(lang, translations["en"])
    if key not in lang_translations:
        print(f"{RED}Missing key: {key} in language {lang}{RESET}")
    return lang_translations.get(key, key)



def is_name_available(NAME):
    if os.getenv("gooberTOKEN"):
        return
    try:
        response = requests.post(f"{VERSION_URL}/check-if-available", json={"name": NAME}, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            data = response.json()
            return data.get("available", False)
        else:
            print(f"{get_translation(LOCALE, 'name_check')}", response.json())
            return False
    except Exception as e:
        print(f"{get_translation(LOCALE, 'name_check2')}", e)
        return False

def register_name(NAME):
    try:
        if ALIVEPING == False:
            return
        # check if the name is avaliable
        if not is_name_available(NAME):
            if os.getenv("gooberTOKEN"):
                return
            print(f"{RED}{get_translation(LOCALE, 'name_taken')}{RESET}")
            quit()
        
        # if it is register it
        response = requests.post(f"{VERSION_URL}/register", json={"name": NAME}, headers={"Content-Type": "application/json"})
        
        if response.status_code == 200:
            data = response.json()
            token = data.get("token")
            
            if not os.getenv("gooberTOKEN"):
                print(f"{GREEN}{get_translation(LOCALE, 'add_token').format(token=token)} gooberTOKEN=<token>.{RESET}")
                quit()
            else:
                print(f"{GREEN}{RESET}")
            
            return token
        else:
            print(f"{RED}{get_translation(LOCALE, 'token_exists').format()}{RESET}", response.json())
            return None
    except Exception as e:
        print(f"{RED}{get_translation(LOCALE, 'registration_error').format()}{RESET}", e)
        return None

register_name(NAME)

class TextGenerator:
    def __init__(self, vocab_size=5000, max_sequence_len=20):
        self.tokenizer = Tokenizer(num_words=vocab_size, lower=True, oov_token="<OOV>")
        self.max_sequence_len = max_sequence_len
        self.vocab_size = vocab_size
        self.model = None
    
    def build_model(self):
        model = Sequential([
            Embedding(self.vocab_size, 128, input_length=self.max_sequence_len - 1),
            Bidirectional(LSTM(256, return_sequences=True)),
            Dropout(0.2),
            LSTM(256),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.vocab_size, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def preprocess_text(self, texts):
        return [text.lower().strip() for text in texts]
    
    def train(self, texts, epochs=20, batch_size=64):
        texts = self.preprocess_text(texts)
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        input_sequences = []
        for sequence in sequences:
            for i in range(1, len(sequence)):
                input_sequences.append(sequence[:i+1])
        
        input_sequences = pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre')
        X, y = input_sequences[:, :-1], input_sequences[:, -1]
        
        self.model = self.build_model()
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True)
        
        self.model.save('text_generator.h5')
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)
    
    def load(self):
        if os.path.exists('text_generator.h5') and os.path.exists('tokenizer.pkl'):
            self.model = load_model('text_generator.h5')
            with open('tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
        else:
            raise FileNotFoundError("Model or tokenizer file not found")
    
    def generate_text(self, seed_text, max_len=20):
        token_list = self.tokenizer.texts_to_sequences([seed_text])
        if not token_list or not token_list[0]:
            return seed_text
        
        token_list = pad_sequences(token_list, maxlen=self.max_sequence_len-1)
        
        generated_text = seed_text
        for _ in range(max_len):
            preds = self.model.predict(token_list, verbose=0)
            next_word = np.argmax(preds, axis=-1)[0]
            word = self.tokenizer.index_word.get(next_word, "")
            if word == "":
                break
            generated_text += " " + word
            token_list = np.append(token_list[:, 1:], [[next_word]], axis=1)
        
        return generated_text
        
text_generator = TextGenerator()

async def run_model_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)

def save_model(self):
    self.model.save('text_generator.h5')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(self.tokenizer, f)

def load_model(filename='text_generator.pkl'):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Model file not found, retraining required.")
        return None

def get_latest_version_info():
    return "Not Avaliable"
    
async def load_cogs_from_folder(bot, folder_name="cogs"):
    for filename in os.listdir(folder_name):
        if filename.endswith(".py") and not filename.startswith("_"):
            cog_name = filename[:-3]
            try:
                await bot.load_extension(f"{folder_name}.{cog_name}")
                print(f"{GREEN}{get_translation(LOCALE, 'loaded_cog')} {cog_name}{RESET}")
            except Exception as e:
                print(f"{RED}{get_translation(LOCALE, 'cog_fail')} {cog_name} {e}{RESET}")
                traceback.print_exc()

currenthash = ""
def generate_sha256_of_current_file():
    global currenthash
    sha256_hash = hashlib.sha256()
    with open(__file__, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    currenthash = sha256_hash.hexdigest()



latest_version = "0.0.0"
local_version = "tensorflow"

def check_for_update():
    print(f"goober Tensorflow edition")
check_for_update()

def get_file_info(file_path):
    try:
        file_size = os.path.getsize(file_path)
        with open(file_path, "r") as f:
            lines = f.readlines()
        return {"file_size_bytes": file_size, "line_count": len(lines)}
    except Exception as e:
        return {"error": str(e)}

nltk.download('punkt')

def load_memory():
    data = []

    # load data from MEMORY_FILE
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        pass

    if not os.path.exists(MEMORY_LOADED_FILE):
        try:
            with open(DEFAULT_DATASET_FILE, "r") as f:
                default_data = json.load(f)
                data.extend(default_data) 
        except FileNotFoundError:
            pass
        with open(MEMORY_LOADED_FILE, "w") as f:
            f.write("Data loaded") 
    return data

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

def train_text_generator(memory, additional_data=None):
    if not memory:
        return None
    text = "\n".join(memory)
    if additional_data:
        text += "\n" + "\n".join(additional_data)
    generator = TextGenerator()
    generator.train(text.split('\n'))
    return generator

#this doesnt work and im extremely pissed and mad
def append_mentions_to_18digit_integer(message):
    pattern = r'\b\d{18}\b'
    return re.sub(pattern, lambda match: f"", message)

def preprocess_message(message):
    message = append_mentions_to_18digit_integer(message)
    tokens = word_tokenize(message)
    tokens = [token for token in tokens if token.isalnum()]
    return " ".join(tokens)


intents = discord.Intents.default()

intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)
memory = load_memory() 
text_generator = load_model()
if not text_generator:
    print(f"{get_translation(LOCALE, 'no_model')}")
    memory = load_memory()
    text_generator = train_text_generator(memory)

generated_sentences = set()
used_words = set()

slash_commands_enabled = False
@bot.event
async def on_ready():
    
    folder_name = "cogs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"{GREEN}{get_translation(LOCALE, 'folder_created').format(folder_name=folder_name)}{RESET}")
    else:
       print(f"{DEBUG}{get_translation(LOCALE, 'folder_exists').format(folder_name=folder_name)}{RESET}")
    text_generator = load_model()
    await load_cogs_from_folder(bot)
    global slash_commands_enabled
    print(f"{GREEN}{get_translation(LOCALE, 'logged_in')} {bot.user}{RESET}")
    try:
        synced = await bot.tree.sync()
        print(f"{GREEN}{get_translation(LOCALE, 'synced_commands')} {len(synced)} {get_translation(LOCALE, 'synced_commands2')} {RESET}")
        slash_commands_enabled = True
        ping_server()
        print(f"{GREEN}{get_translation(LOCALE, 'started').format()}{RESET}")
    except Exception as e:
        print(f"{RED}{get_translation(LOCALE, 'fail_commands_sync')} {e}{RESET}")
        traceback.print_exc()
        quit()
    if not song:
        return  
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name=f"{song}"))

def ping_server():
    if ALIVEPING == "false":
        print(f"{YELLOW}{get_translation(LOCALE, 'pinging_disabled')}{RESET}")
        return
    file_info = get_file_info(MEMORY_FILE)
    payload = {
        "name": NAME,
        "memory_file_info": file_info,
        "version": local_version,
        "slash_commands": slash_commands_enabled,
        "token": gooberTOKEN
    }
    try:
        response = requests.post(VERSION_URL+"/ping", json=payload)
        if response.status_code == 200:
            print(f"{GREEN}{get_translation(LOCALE, 'goober_ping_success').format(NAME=NAME)}{RESET}")
        else:
            print(f"{RED}{get_translation(LOCALE, 'goober_ping_fail')} {response.status_code}{RESET}")
    except Exception as e:
        print(f"{RED}{get_translation(LOCALE, 'goober_ping_fail2')} {str(e)}{RESET}")


positive_gifs = os.getenv("POSITIVE_GIFS").split(',')

def is_positive(sentence):
    scores = analyzer.polarity_scores(sentence)
    sentiment_score = scores['compound']

        # forcin this fucker
    debug_message = f"{DEBUG}{get_translation(LOCALE, 'sentence_positivity')} {sentiment_score}{RESET}"
    print(debug_message) 

    return sentiment_score > 0.1


async def send_message(ctx, message=None, embed=None, file=None, edit=False, message_reference=None):
    if edit and message_reference:
        try:
            # Editing the existing message
            await message_reference.edit(content=message, embed=embed)
        except Exception as e:
            await ctx.send(f"{RED}{get_translation(LOCALE, 'edit_fail')} {e}{RESET}")
    else:
        if hasattr(ctx, "respond"):
            # For slash command contexts
            sent_message = None
            if embed:
                sent_message = await ctx.respond(embed=embed, ephemeral=False)
            elif message:
                sent_message = await ctx.respond(message, ephemeral=False)
            if file:
                sent_message = await ctx.respond(file=file, ephemeral=False)
        else:

            sent_message = None
            if embed:
                sent_message = await ctx.send(embed=embed)
            elif message:
                sent_message = await ctx.send(message)
            if file:
                sent_message = await ctx.send(file=file)
        return sent_message

@bot.hybrid_command(description=f"{get_translation(LOCALE, 'command_desc_retrain')}")
async def retrain(ctx):
    if ctx.author.id != ownerid:
        return

    message_ref = await send_message(ctx, f"{get_translation(LOCALE, 'command_markov_retrain')}")
    try:
        with open(MEMORY_FILE, 'r') as f:
            memory = json.load(f)
    except FileNotFoundError:
        await send_message(ctx, f"{get_translation(LOCALE, 'command_markov_memory_not_found')}")
        return
    except json.JSONDecodeError:
        await send_message(ctx, f"{get_translation(LOCALE, 'command_markov_memory_is_corrupt')}")
        return

    global text_generator
    text_generator = train_text_generator(memory)
    save_model(text_generator)

    await send_message(ctx, f"{get_translation(LOCALE, 'command_markov_retrain_successful').format(data_size=len(memory))}")

@bot.hybrid_command(description=f"{get_translation(LOCALE, 'command_desc_talk')}")
async def talk(ctx, sentence_size: int = 5):
    print("Running text generation")
    if not text_generator:
        await send_message(ctx, f"{get_translation(LOCALE, 'command_talk_insufficent_text')}")
        return
    if sentence_size < 1 or sentence_size > 1000:  # Set reasonable limits
        await send_message(ctx, f"{get_translation(LOCALE, 'command_talk_invalid_length')}")
        return
    try:
        vocab = list(text_generator.tokenizer.word_index.keys())
        if not vocab:
            await send_message(ctx, f"{get_translation(LOCALE, 'command_talk_no_vocab')}")
            return
            
        seed_text = random.choice(vocab)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: text_generator.generate_text(seed_text, sentence_size)
        )
        if not response:
            await send_message(ctx, f"{get_translation(LOCALE, 'command_talk_generation_fail')}")
            return
        cleaned_response = re.sub(r'[^\w\s]', '', response).lower().strip()
        if not cleaned_response:  # Check for empty response after cleaning
            await send_message(ctx, f"{get_translation(LOCALE, 'command_talk_empty_response')}")
            return
        coherent_response = rephrase_for_coherence(cleaned_response)
        if random.random() < 0.9 and is_positive(coherent_response):
            gif_url = random.choice(positive_gifs)
            final_message = f"{coherent_response}\n[jif]({gif_url})"
        else:
            final_message = coherent_response
        await send_message(ctx, final_message[:2000])

    except Exception as e:
        print(f"{RED}Error in talk command: {e}{RESET}")
        traceback.print_exc()
        await send_message(ctx, f"{get_translation(LOCALE, 'command_error')}")

@bot.hybrid_command(description=f"{get_translation(LOCALE, 'command_desc_ask')}")
async def ask(ctx, *, prompt: str):
    if not text_generator:
        await send_message(ctx, f"{get_translation(LOCALE, 'command_ask_insufficient_text')}")
        return

    response = text_generator.generate_text(prompt, 10)

    if response:
        cleaned_response = re.sub(r'[^\w\s]', '', response).lower()
        coherent_response = rephrase_for_coherence(cleaned_response)
        
        if random.random() < 0.9 and is_positive(coherent_response):
            gif_url = random.choice(positive_gifs)
            combined_message = f"{coherent_response}\n[jif]({gif_url})"
        else:
            combined_message = coherent_response

        await send_message(ctx, combined_message)
    else:
        await send_message(ctx, f"{get_translation(LOCALE, 'command_ask_generation_fail')}")



def improve_sentence_coherence(sentence):

    sentence = sentence.replace(" i ", " I ")  
    return sentence

def rephrase_for_coherence(sentence):

    words = sentence.split()

    coherent_sentence = " ".join(words)
    return coherent_sentence

bot.help_command = None


@bot.hybrid_command(description=f"{get_translation(LOCALE, 'command_desc_help')}")
async def help(ctx):
    embed = discord.Embed(
        title=f"{get_translation(LOCALE, 'command_help_embed_title')}",
        description=f"{get_translation(LOCALE, 'command_help_embed_desc')}",
        color=discord.Color.blue()
    )

    command_categories = {
        f"{get_translation(LOCALE, 'command_help_categories_general')}": ["mem", "talk", "about", "ping"],
        f"{get_translation(LOCALE, 'command_help_categories_admin')}": ["stats", "retrain"]
    }

    custom_commands = []
    for cog_name, cog in bot.cogs.items():
        for command in cog.get_commands():
            if command.name not in command_categories[f"{get_translation(LOCALE, 'command_help_categories_general')}"] and command.name not in command_categories[f"{get_translation(LOCALE, 'command_help_categories_admin')}"]:
                custom_commands.append(command.name)

    if custom_commands:
        embed.add_field(name=f"{get_translation(LOCALE, 'command_help_categories_custom')}", value="\n".join([f"{PREFIX}{command}" for command in custom_commands]), inline=False)

    for category, commands_list in command_categories.items():
        commands_in_category = "\n".join([f"{PREFIX}{command}" for command in commands_list])
        embed.add_field(name=category, value=commands_in_category, inline=False)

    await send_message(ctx, embed=embed)

@bot.event
async def on_message(message):
    global memory, markov_model, last_random_talk_time

    if message.author.bot:
        return

    if str(message.author.id) in BLACKLISTED_USERS:
        return

    if message.content.startswith((f"{PREFIX}talk", f"{PREFIX}mem", f"{PREFIX}help", f"{PREFIX}stats", f"{PREFIX}")):
        print(f"{get_translation(LOCALE, 'command_ran').format(message=message)}")
        await bot.process_commands(message)
        return

    if profanity.contains_profanity(message.content):
        return

    if message.content:
        if not USERTRAIN_ENABLED:
            return
        formatted_message = append_mentions_to_18digit_integer(message.content)
        cleaned_message = preprocess_message(formatted_message)
        if cleaned_message:
            memory.append(cleaned_message)
            save_memory(memory)

    # process any commands in the message
    await bot.process_commands(message)

@bot.hybrid_command(description=f"{get_translation(LOCALE, 'command_desc_ping')}")
async def ping(ctx):
    await ctx.defer()
    latency = round(bot.latency * 1000)

    LOLembed = discord.Embed(
        title="Pong!!",
        description=(
            f"{PING_LINE}\n"
            f"`{get_translation(LOCALE, 'command_ping_embed_desc')}: {latency}ms`\n"
        ),
        color=discord.Color.blue()
    )
    LOLembed.set_footer(text=f"{get_translation(LOCALE, 'command_ping_footer')} {ctx.author.name}", icon_url=ctx.author.avatar.url)

    await ctx.send(embed=LOLembed)

@bot.hybrid_command(description=f"{get_translation(LOCALE, 'command_about_desc')}")
async def about(ctx):
    print("-----------------------------------\n\n")
    try:
        check_for_update()
    except Exception as e:
        pass
    print("-----------------------------------")
    embed = discord.Embed(title=f"{get_translation(LOCALE, 'command_about_embed_title')}", description="", color=discord.Color.blue())
    embed.add_field(name=f"{get_translation(LOCALE, 'command_about_embed_field1')}", value=f"{NAME}", inline=False)
    embed.add_field(name=f"{get_translation(LOCALE, 'command_about_embed_field2name')}", value=f"{get_translation(LOCALE, 'command_about_embed_field2value').format(local_version=local_version, latest_version=latest_version)}", inline=False)

    await send_message(ctx, embed=embed)

@bot.hybrid_command(description="stats")
async def stats(ctx):
    if ctx.author.id != ownerid: 
        return
    print("-----------------------------------\n\n")
    try:
        check_for_update()
    except Exception as e:
        pass
    print("-----------------------------------")
    memory_file = 'memory.json'
    file_size = os.path.getsize(memory_file)
    
    with open(memory_file, 'r') as file:
        line_count = sum(1 for _ in file)
    embed = discord.Embed(title=f"{get_translation(LOCALE, 'command_stats_embed_title')}", description=f"{get_translation(LOCALE, 'command_stats_embed_desc')}", color=discord.Color.blue())
    embed.add_field(name=f"{get_translation(LOCALE, 'command_stats_embed_field1name')}", value=f"{get_translation(LOCALE, 'command_stats_embed_field1value').format(file_size=file_size, line_count=line_count)}", inline=False)
    embed.add_field(name=f"{get_translation(LOCALE, 'command_stats_embed_field2name')}", value=f"{get_translation(LOCALE, 'command_stats_embed_field2value').format(local_version=local_version, latest_version=latest_version)}", inline=False)
    embed.add_field(name=f"{get_translation(LOCALE, 'command_stats_embed_field3name')}", value=f"{get_translation(LOCALE, 'command_stats_embed_field3value').format(NAME=NAME, PREFIX=PREFIX, ownerid=ownerid, cooldown_time=cooldown_time, PING_LINE=PING_LINE, showmemenabled=showmemenabled, USERTRAIN_ENABLED=USERTRAIN_ENABLED, last_random_talk_time=last_random_talk_time, song=song, splashtext=splashtext)}", inline=False)
 
    await send_message(ctx, embed=embed)

@bot.hybrid_command()
async def mem(ctx):
    if showmemenabled != "true":
        return
    memory = load_memory()
    memory_text = json.dumps(memory, indent=4)
    with open(MEMORY_FILE, "r") as f:
        await send_message(ctx, file=discord.File(f, MEMORY_FILE))

def improve_sentence_coherence(sentence):
    sentence = sentence.replace(" i ", " I ")
    return sentence

bot.run(TOKEN)