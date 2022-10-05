# CRMExpert
Salesforce CRM Expert Using OpenAI, TensorFlow, Numpy.

# General assumptions:
1. Past chat answers to questions which were persisted to disk are the best answers
2. Load past chat history from disk into memory for speed
3. Return 90% matches as the correct answer without querying OpenAI
4. Use lesser matches as 'few shot' comparisons to increase OpenAI relevance to missing or newer information from when OpenAI stopped crawling the web
5. Use low temperature in OpenAI to reduce the chance of OpenAI just making up false information
6. Save AI responses to disk and memory
7. A human is checking the chat answers JSON files to make sure they are the best answers. Or even better preloading the directory based on a good source such as consultant answers to Email request/response, Slack request/response threads, Salesforce Support Case request/response, etc...

# Future:
## Simultaneous query Google for user question
- Collect Google urls
- Filter out garbage urls, not relevant, have no parseable content, etc...
- Collect text content from urls
- Summarize with OpenAI
- Add to 'few shot' previous knowledge
