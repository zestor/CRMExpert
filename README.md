# CRMExpert
CRM Expert Using OpenAI, TensorFlow Hub, Google Sentence Encoder, Numpy.

# General assumptions:
1. Past chat answers to questions which were persisted to disk are assumed to be groomed into the best answers
2. Load past chat history from disk into memory for speed
3. Return 90% semantically similar matches as the correct answer without querying OpenAI
4. Use lesser matches as 'few shot' examples to increase OpenAI relevance to missing or newer information from when OpenAI stopped crawling the web
5. Use low temperature in OpenAI to reduce the chance of OpenAI just making up false information
6. Save AI responses to disk and memory
7. A human is checking the chat answers JSON files to make sure they are the best answers. Or even better preloading based on a good source such as consultant answers to Email request/response, Slack request/response threads, Salesforce Support Case request/response, etc...

# Future:
## Simultaneous query search engines for user question
- Collect search engine urls
- Filter out garbage urls, not relevant, have no parseable content, etc...
- Collect text content from urls
- Summarize with OpenAI
- Add to 'few shot' previous knowledge
