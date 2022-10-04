import re
import os
import json
import openai
from time import time,sleep
import tensorflow_hub as hub
import numpy as np

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def save_log(payload):
    filename = 'chat/log_%s.json' % time()
    with open(filename, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=1)

def load_logs():
    files = os.listdir('chat/')
    result = list()
    for file in files:
        content = open_file('chat/' + file)
        o = json.loads(content)
        o['file'] = file
        result.append(o)
    return result

#openai.api_key = open_file('openaiapikey.txt')
open_ai_api_key = os.getenv('OPENAI_API_KEY') # not needed, but for clarity

def gpt3_completion(prompt, engine='text-davinci-002', temp=1.1, top_p=1.0, tokens=100, freq_pen=0.0, pres_pen=0.0, stop=['USER:']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return 'GPT3 error: %s' % oops
            print('Error communicating with OpenAI:', oops)
            print(prompt)
            exit()
            sleep(1)

def similar_logs(vector):
    results = list()
    chat = load_logs()
    for i in chat:
        score = np.dot(i['vector'], vector)
        info = i
        info['score'] = score
        print('file: %s score:%d' % (info['file'], info['score']))
        #if score >= 1.0:
        #    continue
        results.append(info)
    ordered = sorted(results, key=lambda d: d['score'], reverse=True)
    try:  # just hack off the ordered list
        ordered = ordered[0:10]
        return ordered
    except:  # if it barfs, send back the whole list because it's too short
        return ordered

if __name__ == '__main__':
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")  # USEv5 is about 100x faster than 4
    print('==============\n\nEXPERT: Hi!')
    similar = list()
    while True:
        # Prompt user for a question
        user_request = input('\nUSER: ')
        if len(user_request) == 0:
            user_request = 'What is the sharing rule limit?'
        # Get vector for user request
        vectors = embed([user_request]).numpy().tolist()
        user_request_vector = vectors[0]
        # Search local knowledge to get few shot for OpenAI prompt
        similar = similar_logs(user_request_vector)

        """
        ai_response = ''
        if len(similar) > 0 and similar[0]['score']>=1:
            ai_response = similar[0]['response']
        else:
            # Create prior knowledge Q&A
            prior_knowledge = ''
            for knowledge in similar:
                prior_knowledge += '\nUSER:%s' % knowledge['request']
                prior_knowledge += '\nEXPERT:%s' % knowledge['response']
            # Create prompt
            prompt = open_file('prompt_marcus.txt').replace('<<USER_REQUEST>>', user_request).replace('<<PRIOR_KNOWLEDGE>>', prior_knowledge)
            # Ask OpenAI
            ai_response = gpt3_completion(prompt, 'text-davinci-002', 0.5)
        """
        ai_response = ''
        # Create prior knowledge Q&A
        prior_knowledge = ''
        for knowledge in similar:
            prior_knowledge += '\nUSER:%s' % knowledge['request']
            prior_knowledge += '\nEXPERT:%s' % knowledge['response']
        # Create prompt
        prompt = open_file('prompt_marcus.txt').replace('<<USER_REQUEST>>', user_request).replace('<<PRIOR_KNOWLEDGE>>', prior_knowledge)
        # Ask OpenAI
        ai_response = gpt3_completion(prompt, 'text-davinci-002', 0.5)

        print('\n', ai_response)
        # Save Q&A to disk
        save_log({'request': user_request, 'vector': user_request_vector, 'response': ai_response})
        
