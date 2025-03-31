'''
author: chris, williams
'''
from enum import Enum
from llms import LLMS

class MODEL_TYPE(Enum):
    ZERO_SHOT = 1
    BREAKDOWN_R = 2
    BREAKDOWN_Q = 3
    SAMPLING = 4

def get_model(model_type, payload, train):
    if model_type == MODEL_TYPE.ZERO_SHOT:
        return ZERO_SHOT_MODEL(payload, train)
    elif model_type == MODEL_TYPE.BREAKDOWN_R:
        return BREAKDOWN_R_MODEL(payload, train)
    elif model_type == MODEL_TYPE.BREAKDOWN_Q:
        return BREAKDOWN_Q_MODEL(payload, train)
    elif model_type == MODEL_TYPE.SAMPLING:
        return SAMPLING_MODEL(payload, train)
    else:
        exit("Invalid model type")

class MODEL:
    Q_TYPE = "Query"

    def __init__(self, payload, train):
        self.request = ""
        self.payload = payload
        self.train = train
        self.llm = LLMS(payload)

    def generate_responses(self, queries, questions, repetitions=10):
        responses = []  # List to store results

        for query, question in zip(queries, questions):
            prompt = f"{self.request}\n{self.Q_TYPE}: {query}"
            response_list = []
            for _ in range(repetitions): # Generate multiple responses
                response = self.llm.infer_messages(prompt)
                if response is not None:
                    response_list.append(response)

            # Store responses in a dictionary
            responses.append({
                self.Q_TYPE: query,
                "question": question,
                "responses": response_list
            })

        self.responses = responses # save a copy
        return responses


class ZERO_SHOT_MODEL(MODEL):
    def __init__(self, payload, train):
        super().__init__(payload, train)
        self.request = """Convert the following SQL query into a natural language question. Your soresponse must be a single sentence in the form of a clear and concise question."""


class BREAKDOWN_MODELS(MODEL):
    def __init__(self, payload, train):
        super().__init__(payload, train)
        self.init_breakdowns()

    def init_breakdowns(self):
        breakdown1 = """Tables Involved: Trip (aliased as T1), Weather (aliased as T2)\nSELECT T1.id: Retrieves the id column from the table T1 (Trip)\nFROM Trip AS T1: Selecting data from the Trip Table, which is aliased as T1\nINNER JOIN weather AS T2 ON T2.zip_code = T1.zip_code: Performs an inner join between the Trip table (T1) and Weather Table, which is aliased as T2, matching rows where the zip_code from both tables are equal\nWHERE T2.min_temperature_f < 45: Filters the rows, keeping only those where the minimum temperature value in the Weather Table (T2) is less than 45"""
        breakdown2 = """Tables Involved: Employee (aliased as T1), Inspection (aliased as T2)\nSELECT CAST(COUNT(DISTINCT T2.inspection_id) AS REAL) / 5: Calculates the distinct count of inspection IDs from the Inspection Table (T2), casts it to REAL, and divides it by 5\nSELECT T1.first_name, T1.last_name: Retrieves the first name and last name columns from the Employee Table (T1)\nFROM Employee AS T1: Selecting data from the Employee Table, which is aliased as T1\nINNER JOIN inspection AS T2 ON T1.employee_id = T2.employee_id: Performs an inner join between the Employee table (T1) and Inspection Table, which is aliased as T2 matching rows where the employee_id from both tables are equal\nWHERE T1.title = 'Sanitarian': Filters the rows to include only those where the title of the employee in the Employee Table (T1) is 'Sanitarian'\nORDER BY T1.salary DESC: Orders the rows by salary in descending order from the Employee Table (T1)\nLIMIT 5: Limits the results to only the first 5 rows based on the salary ordering"""
        breakdown3 = """Tables Involved: Paper (aliased as T1), PaperAuthor (aliased as T2)\nSELECT COUNT(DISTINCT T2.Name): Counts the distinct number of author names from the PaperAuthor Table (T2)\nFROM Paper AS T1: Selecting data from the Paper Table, which is aliased as T1\nINNER JOIN PaperAuthor AS T2 ON T1.Id = T2.PaperId: Performs an inner join between the Paper table (T1) and PaperAuthor Table, which is aliased as T2, matching rows where the PaperId from PaperAuthor Table (T2) is equal to the Id from Paper Table (T1)\nWHERE T1.Title = 'Subcellular localization of nuclease in barley aleurone': Filters the rows to include only those where the title of the paper in the Paper Table (T1) is 'Subcellular localization of nuclease in barley aleurone'"""
        self.breakdowns = [breakdown1, breakdown2, breakdown3]


class BREAKDOWN_R_MODEL(BREAKDOWN_MODELS):
    def __init__(self, payload, train):
        super().__init__(payload, train)
        examples = "\n".join(
            [f"\nSQL Query: {query}\nBreakdown: {breakdown}" for query, breakdown in zip(train['query'][0:3], self.breakdowns)]
        )

        self.request = f"""Break down the following SQL query into its in a human-readable format? Please follow the EXACT breakdown format of the listed examples.
                \nExamples:
                {examples}
                \nNow, breakdown the following SQL query:
                """

class BREAKDOWN_Q_MODEL(BREAKDOWN_MODELS):
    Q_TYPE = "Breakdown"

    def __init__(self, payload, train):
        super().__init__(payload, train)
        examples = "\n".join(
            [f"\nBreakdown: {breakdown}\nQuestion: {question}" for breakdown, question in zip(self.breakdowns, train['question'][0:3])]
        )

        self.request = f"""Convert the following individual steps into a natural language question. Your response must be a single sentence in the form of a clear and concise question that a human with no knowledge of SQL would ask.
                \nExamples:
                {examples}
                \nNow, convert the breakdown into a question. Do not have an intro statement such as "Here is the breakdown" specifying what is happening. Only include the question
                """


class SAMPLING_MODEL(MODEL):
    def __init__(self, payload, train):
        super().__init__(payload, train)
        self.examples = "\n".join(
                [f"\nQuery: {query}\nQuestion: {question}" for query, question in zip(self.train['query'][0:3], self.train['question'][0:3])]
            )

    def set_pos_req(self):
        self.request = f"""Below are examples of a CORRECT query and question pair.
                \nExamples:
                {self.examples}
                \nNow, pick the NUMBER of the following choices that is MOST accurately describes the following SQL query:


                Your response must be a SINGLE NUMBER, no words, no explanations
                """

    def set_neg_req(self):
        self.request = f"""Below are examples of a CORRECT query and question pair.
                \nExamples:
                {self.examples}
                Pick the NUMBER of the following choices that LEAST accurately describes the following SQL query:


                Your response must be a SINGLE NUMBER, no words, no explanations
                """

    def _generate_samples(self, queries, questions, choices_list, samples=40):
        responses = []  # List to store results

        for query, question, choices in zip(queries, questions, choices_list):
            prompt = f"{self.request}\nQuery: {query}\nChoices: {choices}\n"

            response_list = []
            for _ in range(samples): # Generate multiple responses
                response = self.llm.infer_messages(prompt)
                if response is not None:
                    response_list.append(response)

            # Store responses in a dictionary
            responses.append({
                "query": query,
                "question": question,
                "responses": response_list
            })

        self.responses = responses # save a copy
        return responses

    def generate_samples(self, queries, questions, choices_list, samples=40):
        self.set_pos_req()
        self.pos_res = self._generate_samples(queries, questions, choices_list, samples)
        self.set_neg_req()
        self.neg_res = self._generate_samples(queries, questions, choices_list, samples)
        return self.pos_res, self.neg_res
