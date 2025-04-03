'''
author: Krishna
'''
from llms import LLMS, PAYLOADS
from random import random
from sentence_transformers import SentenceTransformer

class SELECTOR:
    def __init__(self, payload=PAYLOADS.FINE_TUNE_8B):
        self.message = """From the provided list of responses, select the single response index that is the most accurate and appropriate.  The query is an SQL query, and the responses are potential natural language questions that could be derived from that query. The responses are provided as a list separated by commans. Choose the best one and only return the index, don't send the wrong index!!"""
        self.llms = LLMS(payload)
        self.sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def _get_cos_similarity(self, correct_question, response_question): #https://www.sbert.net/#usage
        sentences = [correct_question, response_question]
        embeddings = self.sbert_model.encode(sentences)
        similarities = self.sbert_model.similarity(embeddings, embeddings) #sbert defaults to cosine
        return similarities[0][1]

    def pool_responses(self, zero_shot, break_down, sampling):
        responses = []

        zero_shot_dict = {(item['query'], item['question']): item['responses'] for item in zero_shot}
        breakdown_dict = {(item['query'], item['question']): item['responses'] for item in break_down}
        divide_and_conquer_dict = {(item['query'], item['question']): item['responses'] for item in sampling}

        for key in zero_shot_dict:
            if key in breakdown_dict and key in divide_and_conquer_dict:
                responses.append({
                    'query': key[0],
                    'question': key[1],
                    'responses': list(set(zero_shot_dict[key] + breakdown_dict[key] + divide_and_conquer_dict[key])),
                    'index': -1
                })
        
        self.responses = responses

    def select_responses(self, file_name="selection.json"):
        count = 0
        for data in self.responses:
            query = data['query']
            response_list = data['responses']
            count += 1
            prompt = f"{self.message}\nQuery: {query}\nResponses: {response_list}"
            response = self.llms.infer_messages(prompt)
            try:
                response_list[int(response)]
            except:
                response = random.randint(0, len(response_list)-1)

            data['index'] = int(response)

        return self.responses

    def get_accuracy(self):
        inaccuracies = 0
        similarities = []
        for data in self.responses.shape:
            accuracy = self._get_cos_similarity(data['question'], data['responses'][int(data['index'])])
            similarities.append(accuracy)
            if accuracy < 0.7:
                inaccuracies += 1
        return similarities, 1-(inaccuracies/len(self.responses))