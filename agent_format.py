import re
import pandas as pd
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


OLLAMA_URL = "http://127.0.0.1:11434"


def extact_json_from_text(text):
    pattern = r'"s*(\{.*?\}|\[.*?\])\s*"'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        json_string = matches[-1]
        return f"{json_string}"
    else:
        return text


def create_formatting_agent(
    model: str = "llama3.1:8b", ollama_url: str = OLLAMA_URL, temperature: float = 0.0
):
    # LLM with function call
    llm = Ollama(model=model, base_url=ollama_url, temperature=temperature)
    # Prompt
    system = """
    Tu es un grand expert français en Medecine.
    tu peux combiner les mots clès important dans les 6 phrases suivantes en une petite phrase.
    tu dois générer la réponse en français.
    la réponse doit contenir que le résumé sans phrase introductive
    """

    prompt = PromptTemplate(
        template="""{system}.

        Question: {question}
        Réponses Possibles:
        - {possible_answer_a}
        - {possible_answer_b}
        - {possible_answer_c}
        - {possible_answer_d}
        - {possible_answer_e}

        Génération:""",
        input_variables=[
            "question",
            "possible_answer_a",
            "possible_answer_b",
            "possible_answer_c",
            "possible_answer_d",
            "possible_answer_e",
        ],
        partial_variables={"system": system},
    )

    return prompt | llm | StrOutputParser()


if __name__ == "__main__":
    agent = create_formatting_agent()
    questions = pd.read_csv("data/questions.csv", sep=",")
    questions["summary"] = ""

    for idx, row in questions.iterrows():
        summary = agent.invoke(
            {
                "question": row["question"],
                "possible_answer_a": row["answer_A"],
                "possible_answer_b": row["answer_B"],
                "possible_answer_c": row["answer_C"],
                "possible_answer_d": row["answer_D"],
                "possible_answer_e": row["answer_E"],
            }
        )

        questions.at[idx, "summary"] = summary
        questions.to_csv("data/questions_with_summary.csv")
