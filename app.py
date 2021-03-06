#!/usr/bin/env python
# coding: utf-8

# In[32]:


from flask import Flask, render_template, request
app = Flask(__name__)


# In[33]:


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def init_model_tokenizer():
    model_name="microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_response(model, tokenizer, chat_round, chat_history_ids, text):
    new_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_round > 0 else new_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    r = "Chatbot Reply: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
    return chat_history_ids, r

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method=="POST":
        text = str(request.form.get("text"))
        chat_history_ids = None
        model, tokenizer = init_model_tokenizer()
        for chat_round in range(1):
            
            chat_history_ids, r = generate_response(model, tokenizer, chat_round, chat_history_ids, text)
        return(render_template("index.html", result = r))
    else:
        return(render_template("index.html", result = "Waiting..."))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




