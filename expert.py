#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import random
from datetime import date
from time import time

import numpy as np
import tensorflow_hub as hub

from python.ZestorHelper import ZestorHelper


# persist to memory and disk
def save_chat(prior_chat, payload):
    ZestorHelper.mkdir_if_not_exists('./chat')
    filename = '%s.json' % time()
    # persist to disk
    with open('./chat/' + filename, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=1)
    # add node to prior_chat
    payload['file'] = filename
    prior_chat.append(payload)
    # return appended chat
    return prior_chat

# load prior chat knowledge from disk
# assumes answers persisted to disk are best answer
def load_prior_chat():
    retval = list()
    for file in os.listdir('./chat/'):
        chat_file_content = ZestorHelper.open_file('./chat/' + file)
        chat_json = json.loads(chat_file_content)
        chat_json['file'] = file
        retval.append(chat_json)
    return retval

# search for similar questions
# and build out knowledge of answers
# assumes answers persisted to disk are best answer
def search_prior_chat(prior_chat, vector):
    retval = list()
    for chat in prior_chat:
        similarity_score = np.dot(chat['vector'], vector)
        temp_chat = chat
        temp_chat['rank'] = similarity_score
        # only return 40% or greater matches
        if similarity_score >= 0.4:
            print('EXPERT INTERNAL KNOWLEDGE: %d percent match to prior knowledge %s %s' % (similarity_score * 100, temp_chat['request'], temp_chat['file']))
            retval.append(temp_chat)
    retval.sort(key=lambda d:d['rank'], reverse=True)
    if len(retval) > 5:
        retval = retval[0:5]
    return retval

# Main program
if __name__ == '__main__':

    # Google Universal Sentence Encode v5
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    print('==============\n\nEXPERT: Hi!')
    semantic_similar_searches = list()

    # load prior chat knowledge into memory
    prior_chat = load_prior_chat()

    while True:
        # Prompt user for a question
        user_request = input('\nUSER: ')
        if len(user_request) == 0:
            user_request = 'What is the sharing rule limit?'
        
        # Get vector for user request
        vectors = embed([user_request]).numpy().tolist()
        user_request_vector = vectors[0]
        
        # Search local knowledge to get few shot for OpenAI prompt
        semantic_similar_searches = search_prior_chat(prior_chat, user_request_vector)

        ai_response = ''
        ai_references = ''
        if len(semantic_similar_searches) > 0 and semantic_similar_searches[0]['rank'] > 0.9:
            foo = ['As I mentioned before, ', 'From our prior discussion, ', 'As I previously mentioned, ']
            ai_response = random.choice(foo)
            ai_response += semantic_similar_searches[0]['response']
        else:
            # Create prior knowledge Q&A
            prior_knowledge = ''
            for knowledge in semantic_similar_searches:
                prior_knowledge += '\nUSER:%s' % knowledge['request']
                prior_knowledge += '\nEXPERT:%s' % knowledge['response']
                prior_knowledge = prior_knowledge.strip()

            #print('Prior knowledge: %s' % prior_knowledge)
            prior_knowledge += '\nUSER:%s' % user_request
            prior_knowledge = prior_knowledge.strip()
            prior_knowledge += '\nEXPERT:'

            # Create prompt
            prompt = ZestorHelper.open_file('./promptTemplates/expert.txt').replace('<<USER_REQUEST>>', prior_knowledge)
            
            # Ask OpenAI
            ai_response = ZestorHelper.openai_callout_noretry(prompt, 'text-davinci-002', 0.5)

            # Get Url References
            prompt = 'Given the following article, List URL references:\r\nArticle:<<CONTENT>>\r\nOutput:\r\n'.replace('<<CONTENT>>', ai_response)

            # Ask OpenAI
            ai_references = ZestorHelper.openai_callout_noretry(prompt, 'text-davinci-002', 0.5)

            # Save Q&A to disk
            prior_chat = save_chat(prior_chat, {'date': str(date.today()), 'request': user_request, 'vector': user_request_vector, 'response': ai_response, 'references': ai_references})
        print('\nEXPERT:' +  ai_response + '\r\nReferences:' + ai_references)


        
