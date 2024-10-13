import os
import re
import time
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI


MODEL_NAME = "mistral-large-latest"
MISTRAL_API_KEY = "***"
TEMPERATURE = 0.5

os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY
LLM = ChatMistralAI(model_name=MODEL_NAME, temperature=TEMPERATURE)


def process_answer(answer: str) -> str:
    # match answer
    match = re.search(r"(?:^|\n)([A-E](?:,[A-E])*)(?:\n|$)", answer)
    if match:
        answer = match.group(1)
    else:
        answer = answer.strip()

    # extract valid answer choices
    choices = re.findall(r"[A-E]", answer)
    if not choices:
        return ""

    unique_sorted_choices = sorted(set(choices))
    return ",".join(unique_sorted_choices)


def agent_generate_answer():
    template = """
    You are a world-class highly specialized French medical expert, tasked with answering exam questions on French Medical Practice. Each question may have multiple correct answers, and it is crucial to be precise and accurate.
    
    Use both the provided **context** and your own **medical expertise** to ensure that your answers are correct. If the context provides specific information, prioritize that. However, you may also rely on your broad medical knowledge to supplement where necessary.

    ---
    ## Question
    {question}

    ## Multiple Choice Answers
    A: {answer_A}
    B: {answer_B}
    C: {answer_C}
    D: {answer_D}
    E: {answer_E}

    ---
    ## Context
    {context}

    ---
    ## Expected Output Format
    Only provide the letters for the correct answers, alphabetically sorted and separated by commas without space (e.g., "A,B,C" or "C"). 
        
    **IMPORTANT**: 
    - Ensure no other output except the letters is generated.
    - Be as accurate as possible, using both your knowledge and the provided context.
    
    -> **BONUS**: I will give a huge amount of money for the perfect answer.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "question",
            "answer_A",
            "answer_B",
            "answer_C",
            "answer_D",
            "answer_E",
            "context",
        ],
    )

    answer_generator = prompt | LLM
    return answer_generator


def agent_generate_answer_reflect():
    template = """
    You are a world-class, highly specialized French medical expert tasked with answering exam questions on French Medical Practice. Each question may have multiple correct answers. Your task is to provide the most accurate answers possible, ensuring precision and thorough analysis.

    Follow these steps:

    1. Begin by enclosing all thoughts within <thinking> tags. Analyze the question thoroughly and explore all potential interpretations, angles, and approaches.
    2. Consider each answer option (A, B, C, D, E) carefully. Discuss its merits and weaknesses in detail using both the **context** and your own **medical expertise**. If uncertain, explain why, and reason out the most plausible answer using evidence-based reasoning.
    3. If the context provides specific information, prioritize that. Where necessary, supplement with general medical knowledge and official guidelines.
    4. After evaluating all options, provide a final answer inside the <output> tags. 

    Regularly assess your reasoning by using <reflection> tags for intermediate evaluations. Adjust your approach as needed.

    ---
    ## Question
    {question}

    ## Multiple Choice Answers
    A: {answer_A}
    B: {answer_B}
    C: {answer_C}
    D: {answer_D}
    E: {answer_E}

    ---
    ## Context
    {context}

    ## Expected Output Format
    Return the final answers for the question inside <output> and </output> tags. Provide a detailed explanation for your final answer, addressing why you chose it and why others were rejected (if applicable).

    **IMPORTANT**: 
    - Be as accurate as possible. Rely on both the context and your deep medical expertise.
    - Acknowledge uncertainties when necessary and use logical reasoning to choose the most plausible answer.

    -> **BONUS**: The perfect answer will be rewarded with a huge amount of money.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "question",
            "answer_A",
            "answer_B",
            "answer_C",
            "answer_D",
            "answer_E",
            "context",
        ],
    )

    answer_generator = prompt | LLM
    return answer_generator


def agent_review_answer():
    template = """
    You are a world-class highly specialized French medical expert, tasked with ctritically reviewing the submitted answers for exam questions on French Medical Practice. Each question may have multiple correct answers, and it is crucial to be precise and accurate.
    
    Use both the provided **context** and your own **medical expertise** to properly review the sumbitted answer. Provide the final gold answer (100% correct).

    You are a world-class, highly specialized French medical expert, tasked with **critically reviewing** submitted answers for exam questions on French Medical Practice. Each question may have multiple correct answers, and it is crucial to ensure 100% accuracy.

    Your task is to:
    
    1. Carefully compare the submitted answer with the provided **context** and your own **medical expertise**.
    2. Explicitly check for errors, incorrect assumptions, or misinterpretations in the reasoning. Reflect on why the submitted answer could be wrong, and correct any inaccuracies.
    3. Ensure that the final answer is based on both evidence from the context and established medical knowledge.

    ---
    ## Question
    {question}

    ## Multiple Choice Answers
    A: {answer_A}
    B: {answer_B}
    C: {answer_C}
    D: {answer_D}
    E: {answer_E}

    ---
    ## Sumbitted Answer
    {generation}

    ---
    ## Context
    {context}

    ---
    ## Expected Output Format
    Only provide the **letters of the correct answers**, alphabetically sorted and separated by commas without spaces (e.g., "A,B,C" or "C").

    **IMPORTANT**: 
    - Ensure that no other output is generated except the letters of the correct answers.
    - Be as accurate as possible, using both your medical knowledge and the provided context.

    -> **BONUS**: The perfect answer will be rewarded with a huge amount of money.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "question",
            "answer_A",
            "answer_B",
            "answer_C",
            "answer_D",
            "answer_E",
            "context",
            "generation",
        ],
    )

    answer_generator = prompt | LLM
    return answer_generator


def generate_answers(question_filepath, output_filepath, sleep_time: int = 3):
    df = pd.read_csv(question_filepath, sep=",")
    answer_generator = agent_generate_answer()
    answers = []
    for row_idx, row in df.iterrows():
        while True:
            try:
                time.sleep(sleep_time)
                response = answer_generator.invoke(
                    {
                        "question": row["question"],
                        "answer_A": row["answer_A"],
                        "answer_B": row["answer_B"],
                        "answer_C": row["answer_C"],
                        "answer_D": row["answer_D"],
                        "answer_E": row["answer_E"],
                        "context": row["context"],
                    }
                )
                answer = process_answer(response.content)
                if not answer and TEMPERATURE != 0:
                    # retry
                    raise Exception("Invalid answer")
                print(f"Question {row_idx + 1}: {answer}")
                answers.append(answer)
                break  # only exit if successful
            except Exception as e:
                print(f"Error occurred for question {row_idx + 1}: {e}. \nRetrying...")

    # save answers
    output_df = pd.DataFrame(answers, columns=["Answer"])
    output_df.index.name = "id"
    output_df.to_csv(output_filepath)


def generate_answers_agentic(
    question_filepath, output_filepath, complete_output_filepath, sleep_time: int = 3
):
    df = pd.read_csv(question_filepath, sep=",")

    # load agents
    answer_generator = agent_generate_answer_reflect()
    answer_reviewer = agent_review_answer()

    final_answers = []
    complete_data = []
    total_questions = len(df)
    for row_idx, row in df.iterrows():
        print(
            f"================== Progress Q {row_idx + 1}/{total_questions} =================="
        )
        # generate answer
        while True:
            try:
                time.sleep(sleep_time)
                generated_answer = answer_generator.invoke(
                    {
                        "question": row["question"],
                        "answer_A": row["answer_A"],
                        "answer_B": row["answer_B"],
                        "answer_C": row["answer_C"],
                        "answer_D": row["answer_D"],
                        "answer_E": row["answer_E"],
                        "context": row["context"],
                    }
                )
                generated_answer = generated_answer.content.strip()
                if not generated_answer:
                    raise Exception("Invalid generated answer.")
                print(
                    f"Generated answer for Question {row_idx + 1}: {generated_answer}"
                )
                break  # only exit if successful
            except Exception as e:
                print(
                    f"Error in generation for Question {row_idx + 1}: {e}. \nRetrying..."
                )

        # review answer
        while True:
            try:
                time.sleep(sleep_time)
                review_answer = answer_reviewer.invoke(
                    {
                        "question": row["question"],
                        "answer_A": row["answer_A"],
                        "answer_B": row["answer_B"],
                        "answer_C": row["answer_C"],
                        "answer_D": row["answer_D"],
                        "answer_E": row["answer_E"],
                        "context": row["context"],
                        "generation": generated_answer,
                    }
                )
                review_answer = review_answer.content.strip()
                final_answer = process_answer(review_answer)
                if not final_answer:
                    raise Exception("Invalid final answer.")
                print(f"Final answer for Question {row_idx + 1}: {final_answer}")
                break  # only exit if successful
            except Exception as e:
                print(f"Error in review for Question {row_idx + 1}: {e}. \nRetrying...")

        final_answers.append(final_answer)

        complete_row = row.to_dict()
        complete_row["generated_answer"] = generated_answer
        complete_row["final_answer"] = final_answer
        complete_data.append(complete_row)

        # Save the complete results (for analysis)
        complete_df = pd.DataFrame(complete_data)
        complete_df.to_csv(complete_output_filepath, index=False)

        # Save only the final answers (for sumbission)
        final_answer_df = pd.DataFrame(final_answers, columns=["Answer"])
        final_answer_df.index.name = "id"
        final_answer_df.to_csv(output_filepath)

    # # Save the complete results (for analysis)
    # complete_df = pd.DataFrame(complete_data)
    # complete_df.to_csv(complete_output_filepath, index=False)

    # # Save only the final answers (for sumbission)
    # final_answer_df = pd.DataFrame(final_answers, columns=["Answer"])
    # final_answer_df.index.name = "id"
    # final_answer_df.to_csv(output_filepath)


if __name__ == "__main__":
    # generate_answers(
    #     question_filepath="/home/outscale/repos/dr.ai/data/questions_with_contexts.csv",
    #     output_filepath="data/final_answers.csv",
    #     sleep_time=3
    # )

    generate_answers_agentic(
        question_filepath="/home/outscale/repos/dr.ai/data/questions_with_contexts_last_fixed.csv",
        output_filepath="data/final_answers.csv",
        complete_output_filepath="data/complete_answers.csv",
        sleep_time=5,
    )
