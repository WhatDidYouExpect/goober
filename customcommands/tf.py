import discord
from discord.ext import commands
import os
from typing import List, TypedDict
import numpy as np
import json
from time import strftime, localtime
import pickle
import functools
import re
import time
import asyncio

ready: bool = True
MODEL_MATCH_STRING = "[0-9]{2}_[0-9]{2}_[0-9]{4}-[0-9]{2}_[0-9]{2}"

try:
    import tensorflow as tf    
    from tensorflow import keras
    from keras.preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Embedding, LSTM, Dense
    from keras.models import load_model
    from keras.backend import clear_session
    tf.config.optimizer.set_jit(True)
except ImportError:
    print("ERROR: Failed to import Tensorflow. Here is a list of required dependencies:",(
        "tensorflow==2.10.0"
        "(for Nvidia users: tensorflow-gpu==2.10.0)"
        "(for macOS: tensorflow-metal==0.6.0, tensorflow-macos==2.10.0)"
        "numpy~=1.23"
    ))
    ready = False

class TFCallback(keras.callbacks.Callback):
    def __init__(self,bot, progress_embed:discord.Embed, message):
        self.embed:discord.Embed = progress_embed
        self.bot:commands.Bot = bot
        self.message = message
        self.times:List[int] = [time.time()]
        
    def on_train_begin(self, logs=None):
        pass
        
    async def send_message(self,message:str, description:str, **kwargs):
        if "epoch" in kwargs:
            self.times.append(time.time())
            average_epoch_time:int = np.average(np.diff(np.array(self.times)))
            description = f"ETA: {round(average_epoch_time)}s"
        self.embed.add_field(name=f"<t:{round(time.time())}:t> - {message}",value=description,inline=False)
        await self.message.edit(embed=self.embed)
    
    def on_train_end(self,logs=None):
        self.bot.loop.create_task(self.send_message("Training stopped", "training has been stopped."))
        
    def on_epoch_begin(self, epoch, logs=None):
        self.bot.loop.create_task(self.send_message(f"Starting epoch {epoch}","This might take a while", epoch=True))
        
    def on_epoch_end(self, epoch, logs=None):
        self.bot.loop.create_task(self.send_message(f"Epoch {epoch} ended",f"Accuracy: {round(logs.get('accuracy',0.0),4)}"))
        

class Ai:
    def __init__(self):
        model_path = settings.get("model_path")
        if model_path:
            self.__load_model(model_path)
        self.is_loaded = model_path is not None
        self.batch_size = 64 
        
    def get_model_name_from_path(self,path:str):
        match:re.Match = re.search(MODEL_MATCH_STRING, path)
        return path[match.start():][:match.end()]

    def generate_model_name(self) -> str:
        return strftime('%d_%m_%Y-%H_%M', localtime())
    
    def generate_model_abs_path(self, name:str):
        name = name or self.generate_model_name()
        return os.path.join(".","models",self.generate_model_name(),"model.h5")

    def generate_tokenizer_abs_path(self, name:str):
        name = name or self.generate_model_name()
        return os.path.join(".","models",name,"tokenizer.pkl")
        
    def generate_info_abs_path(self,name:str):
        name = name or self.generate_model_name()
        return os.path.join(".","models",name,"info.json")

        
    def save_model(self,model, tokenizer, history, _name:str=None):
        name:str = _name or self.generate_model_name()
        os.makedirs(os.path.join(".","models",name), exist_ok=True)
        
        with open(self.generate_info_abs_path(name),"w") as f:
            json.dump(history.history,f)
        
        with open(self.generate_tokenizer_abs_path(name), "wb") as f:
            pickle.dump(tokenizer,f)
        
        model.save(self.generate_model_abs_path(name))

        
    def __load_model(self, model_path:str):
        clear_session()
        self.model = load_model(os.path.join(model_path,"model.h5")) 
        
        model_name:str = self.get_model_name_from_path(model_path)
        
        try:
            with open(self.generate_tokenizer_abs_path(model_name),"rb") as f:
                self.tokenizer = pickle.load(f)
        except FileNotFoundError:
            print("Failed to load tokenizer for model... Using default")
            self.tokenizer = Tokenizer()
            
            with open("memory.json","r") as f:
                self.tokenizer.fit_on_sequences(json.load(f))
        self.is_loaded = True

    def reload_model(self):
        clear_session()
        model_path:str = settings.get("model_path")
        if model_path:
            self.model = self.__load_model(model_path)
            self.is_loaded = True
    
    async def run_async(self,func,bot,*args,**kwargs):
        func = functools.partial(func,*args,**kwargs)
        return await bot.loop.run_in_executor(None,func)
            
class Learning(Ai):
    def __init__(self):
        super().__init__()
        
    def __generate_labels_and_inputs(self,memory: List[str], tokenizer=None) -> tuple:
        if not tokenizer:
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(memory)
        sequences = tokenizer.texts_to_sequences(memory)
        
        x = []
        y = []
        for seq in sequences:
            for i in range(1, len(seq)):
                x.append(seq[:i])
                y.append(seq[i])
                
        return x,y, tokenizer
    
    def create_model(self,memory: list, iters:int=2):
        memory = memory[:2000]
        X,y,tokenizer = self.__generate_labels_and_inputs(memory)
        maxlen:int = max([len(x) for x in X]) 
        x_pad = pad_sequences(X, maxlen=maxlen, padding="pre")
        
        y = np.array(y)
        
        model = Sequential()
        model.add(Embedding(input_dim=VOCAB_SIZE,output_dim=128,input_length=maxlen))
        model.add(LSTM(64))
        model.add(Dense(VOCAB_SIZE, activation="softmax"))
        
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(x_pad, y, epochs=iters, batch_size=64, callbacks=[tf_callback])
        self.save_model(model, tokenizer, history)
        return

        
    def add_training(self,memory: List[str], iters:int=2):
        tokenizer_path = os.path.join(settings.get("model_path"),"tokenizer.pkl")
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
            
        X,y,_ = self.__generate_labels_and_inputs(memory, tokenizer)
        
        maxlen:int = max([len(x) for x in X]) 
        x_pad = pad_sequences(X, maxlen=maxlen, padding="pre")
        y = np.array(y)
        
        history = self.model.fit(x_pad,y, epochs=iters, validation_data=(x_pad,y), batch_size=64, callbacks=[tf_callback]) # Ideally, validation data would be seperate from the actual data
        self.save_model(self.model,tokenizer,history,self.get_model_name_from_path(settings.get("model_path")))
        return
    
class Generation(Ai):
    def __init__(self):
        super().__init__()
        
    def generate_sentence(self, word_amount:int, seed:str):
        if not self.is_loaded:
            return False
        for _ in range(word_amount):
            token_list = self.tokenizer.texts_to_sequences([seed])[0]
            token_list = pad_sequences([token_list], maxlen=self.model.layers[0].input_shape[1], padding="pre")

            output_word = "" # Sometimes model fails to predict the word, so using a fallback 

            predicted_probs = self.model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]

            for word, index in self.tokenizer.word_index.items():
                if index == predicted_word_index:
                    output_word = word
                    break

            seed += " " + output_word
        return seed
        
    
VOCAB_SIZE = 100_000
SETTINGS_TYPE = TypedDict("SETTINGS_TYPE", {
    "model_path":str, # path to the base folder of the model, aka .../models/05-01-2025-22_31/
    "tokenizer_path":str,
})

tf_callback:TFCallback
model_dropdown_items = []
settings: SETTINGS_TYPE = {}

target_message:int
learning:Learning
generation: Generation

class Settings:
    def __init__(self):
        self.settings_path:str = os.path.join(".","models","settings.json")
        
    def load(self):
        global settings
        try:
            with open(self.settings_path,"r") as f:
                settings = json.load(f)
        except FileNotFoundError:
            with open(self.settings_path,"w") as f:
                json.dump({},f)

    def change_model(self,new_model_base_path:str):
        global settings
        new_model_path = os.path.join(".","models",new_model_base_path)
                
        with open(self.settings_path,"r") as f:
            settings = json.load(f)
            
        settings["model_path"] = new_model_path
        
        with open(self.settings_path, "w") as f:
            json.dump(settings,f)


class Dropdown(discord.ui.Select):
    def __init__(self, items:List[str]):
        global model_dropdown_items
        model_dropdown_items = []
        
        for item in items:
            model_dropdown_items.append(
                discord.SelectOption(label=item)
            )
        
        super().__init__(placeholder="Select model", options=model_dropdown_items)
        
    async def callback(self, interaction: discord.Interaction):
        if int(interaction.user.id) != int(os.getenv("ownerid")):
            await interaction.message.channel.send("KILL YOURSELF")
        Settings().change_model(self.values[0])
        await interaction.message.channel.send(f"Changed model to {self.values[0]}")
    
class DropdownView(discord.ui.View):
    def __init__(self, timeout, models):
        super().__init__(timeout=timeout)
        self.add_item(Dropdown(models))


class Tf(commands.Cog):    
    def __init__(self,bot):
        global learning, generation, ready
        os.makedirs(os.path.join(".","models"),exist_ok=True)
        Settings().load()
        self.bot = bot
        learning = Learning()
        generation = Generation()
    
    @commands.command()
    async def start(self,ctx):
        await ctx.defer()
        await ctx.send("hi")
        
    @commands.command()
    async def generate(self,ctx,seed:str,word_amount:int=5):
        await ctx.defer()
        await ctx.send(generation.generate_sentence(word_amount,seed))
    
    @commands.command()
    async def create(self,ctx:commands.Context, epochs:int=3):
        global tf_callback
        await ctx.defer()
        with open("memory.json","r") as f:
            memory:List[str] = json.load(f)
        await ctx.send("Initializing tensorflow")
        embed = discord.Embed(title="Creating a model...", description="Progress of creating a model")
        embed.set_footer(text="Note: Progress tracking might report delayed / wrong data, since the function is run asynchronously")
        target_message:discord.Message = await ctx.send(embed=embed)
        
        tf_callback = TFCallback(self.bot,embed,target_message)
        await learning.run_async(learning.create_model,self.bot,memory,epochs)
        embed = target_message.embeds[0]
        embed.add_field(name=f"<t:{round(time.time())}:t> Finished",value="Model saved.")
        await target_message.edit(embed=embed)
        
        
    @commands.command()
    async def train(self,ctx, epochs:int=2):
        global tf_callback
        
        await ctx.defer()
        with open("memory.json","r") as f:
            memory:List[str] = json.load(f)
            
        embed = discord.Embed(title="Training model...", description="Progress of training model")
        target_message = await ctx.send(embed=embed)
        tf_callback = TFCallback(self.bot,embed,target_message)

        await learning.run_async(learning.add_training,self.bot,memory,epochs)
        await ctx.send("Finished!")
    
    @commands.command() 
    async def change(self,ctx,model:str=None):
        embed = discord.Embed(title="Change model",description="Which model would you like to use?")
        if model is None:
            models:List[str] = os.listdir(os.path.join(".","models"))
            models = [folder for folder in models if re.match(MODEL_MATCH_STRING,folder)]
            if len(models) == 0:
                models = ["No models available."]
            await ctx.send(embed=embed,view=DropdownView(90,models))
            learning.reload_model()
            generation.reload_model()

async def setup(bot):
    await bot.add_cog(Tf(bot))
