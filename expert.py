#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import re
from time import sleep, time

import numpy as np
import tensorflow_hub as hub

from python.ZestorHelper import ZestorHelper

def save_chat(payload):
    ZestorHelper.mkdir_if_not_exists('./chat')
    filename = './chat/%s.json' % time()
    with open(filename, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=1)

def load_prior_chat():
    files = os.listdir('chat/')
    result = list()
    for file in files:
        content = ZestorHelper.open_file('./chat/' + file)
        o = json.loads(content)
        o['file'] = file
        result.append(o)
    return result

def search_prior_chat(vector):
    results = list()
    chat = load_prior_chat()
    for i in chat:
        score = np.dot(i['vector'], vector)
        info = i
        info['score'] = score
        print('file: %s score:%d' % (info['file'], info['score']))
        results.append(info)
    reordered_list = sorted(results, key=lambda d: d['score'], reverse=True)
    try:
        # first 10 items
        reordered_list = reordered_list[0:10]
        return reordered_list
    except:
        return reordered_list

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
        similar = search_prior_chat(user_request_vector)

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
        prompt = ZestorHelper.open_file('./promptTemplates/expert.txt').replace('<<USER_REQUEST>>', user_request).replace('<<PRIOR_KNOWLEDGE>>', prior_knowledge)
        # Ask OpenAI
        ai_response = ZestorHelper.openai_callout(prompt, 'text-davinci-002', 0.5)

        print('\n', ai_response)
        # Save Q&A to disk
        save_chat({'request': user_request, 'vector': user_request_vector, 'response': ai_response})
        
