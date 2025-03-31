'''
author: williams
'''
from training import BREAKDOWN_Q_MODEL, SAMPLING_MODEL
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import numpy


class SAMPLING:
    def __init__(self, payload, train, noQueries):
        self.bq_model = BREAKDOWN_Q_MODEL(payload, train)
        self.s_model = SAMPLING_MODEL(payload, train)
        self.noQueries = noQueries
        self.sbertModel = SentenceTransformer('all-MiniLM-L6-v2')
        self.useModel = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def get_cos_similarity(self, correct_question, response_question):
        correct_embedding = self.useModel([correct_question]).numpy()
        response_embedding = self.useModel([response_question]).numpy()

        cosine_sim = cosine_similarity(correct_embedding, response_embedding)
        return cosine_sim[0][0]

    def choose_best_difference(self, pos_samples, neg_samples,length):
        pos_int_array = [int(num) for num in pos_samples if num is not None]
        neg_int_array = [int(num) for num in neg_samples if num is not None]

        diff_array = numpy.zeros(length)

        for i in range(len(pos_int_array)):
            diff_array[pos_int_array[i]] += 1

        for i in range(len(neg_int_array)):
            diff_array[neg_int_array[i]] -= 1

        return numpy.argmax(diff_array)

    def is_valid_integer(self, num):
        try:
            int(num)
            return True
        except (ValueError, TypeError):
            return False

    def choose_best_difference_debug(self, pos_samples,neg_samples,length):
        pos_int_array = [int(num) - 1 for num in pos_samples if num is not None and self.is_valid_integer(num)]
        neg_int_array = [int(num) - 1 for num in neg_samples if num is not None and self.is_valid_integer(num)]

        diff_array = numpy.zeros(length)
        pos_count = numpy.zeros(length)
        neg_count = numpy.zeros(length)

        for i in range(len(pos_int_array)):
            pos_count[pos_int_array[i]] += 1
            diff_array[pos_int_array[i]] += 1

        for i in range(len(neg_int_array)):
            neg_count[neg_int_array[i]] += 1
            diff_array[neg_int_array[i]] -= 1

        return numpy.argmax(diff_array), numpy.array(pos_count), numpy.array(neg_count)


    def generate_samples(self, queries, questions, repetitions=4, samples=40):
        self.bq_model.generate_responses(queries, questions, repetitions)
        self.choices_list = [d.get('responses') for d in self.bq_model.responses[0:self.noQueries]]
        choices_formatted = [
            "\n".join([f"{index}: {string}" for index, string in enumerate(inner_list, 1)])
            for inner_list in self.choices_list
        ]
        self.s_model.generate_samples(queries, questions, choices_formatted, samples)
        self.count = len(self.s_model.pos_res)

    def compare_responses(self, save_png = False):
        for i in range(self.count):
            pos_samples = self.s_model.pos_res[i].get('responses')
            neg_samples = self.s_model.neg_res[i].get('responses')
            argMax, pos_count, neg_count = self.choose_best_difference_debug(pos_samples, neg_samples, len(self.choices_list[i]))
            accurate_question = self.bq_model.responses[i].get('question')
            response_question = self.bq_model.responses[i].get('responses')[argMax]

            similarity_score = numpy.zeros(len(self.choices_list[i]))
            for j in range(len(self.choices_list[i])):
                responseI = self.bq_model.responses[i].get('responses')[j]
                similarity_score[j] = f"{round(self.get_cos_similarity(accurate_question, responseI), 4):.4f}"

            sim_string = [str(num) for num in similarity_score]
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.bar(sim_string, pos_count)
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Times Chosen')
            plt.title('Best Candidate')
            plt.ylim(0, 40)

            plt.subplot(1, 2, 2)
            plt.bar(sim_string, neg_count)
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Times Chosen')
            plt.title('Worst Candidate')
            plt.ylim(0, 40)

            plt.tight_layout()
            plt.show()

            if save_png:
                plt.savefig(f"output_{i}.png")

            #print(few_shot_breakdown_questions_candid[i].get('responses'))
            print("Best Choice:", argMax + 1, "", response_question)
            print("\n\n")
