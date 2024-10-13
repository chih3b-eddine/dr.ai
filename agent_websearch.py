import os
import re
import requests
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.ollama import Ollama
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate

os.environ["TAVILY_API_KEY"] = "tvly-***"
DOMAINS = [
    "https://www.cours-medecine.info/index.html",
    "https://wikimedi.ca/",
    "https://www.pedia-univ.fr/",
    "https://www.msdmanuals.com/",
    "https://www.medg.fr/",
]
OLLAMA_URL = "http://127.0.0.1:11434"


def parse_url(url: str):
    try:
        html = requests.get(url).content
        text = BeautifulSoup(html, features="html.parser").get_text()
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"[\r\t\f\v ]+", " ", text)
        if "�" in text:
            text = ""
    except Exception as e:
        text = ""
    return text


def create_webpage_summary_agent(
    model: str = "llama3.1:8b", ollama_url: str = OLLAMA_URL, temperature: float = 0.2
):
    llm = Ollama(model=model, base_url=ollama_url, temperature=temperature)
    system = """Tu es un grand expert français en Medecine."""
    prompt = PromptTemplate(
        template="""{system}.
        # Instruction: Rédiges un résumé en français du document ci-dessous
        Concentre-toi sur les informations pertinentes à la question: "{question}"

        # Document à résumer:
        {document}
        """,
        input_variables=["question", "document"],
        partial_variables={"system": system},
    )
    return prompt | llm | StrOutputParser()


def create_websearch_agent(max_results: int = 1, include_domains: list[str] = []):
    websearch_tool = TavilySearchResults(
        max_results=max_results,
        search_depth="advanced",
        include_domains=include_domains,
    )

    def parse_output(search_result):
        urls = [item["url"] for item in search_result]
        output = []
        for url in urls:
            output.append(parse_url(url))
        return dict(context="\n\n\n\n".join(output))

    websearch_agent = websearch_tool | parse_output
    return websearch_agent


if __name__ == "__main__":
    websearch_agent = create_websearch_agent(max_results=2, include_domains=[])
    summary_agent = create_webpage_summary_agent()

    questions = pd.read_csv("/data/questions_with_summary.csv")
    questions["context"] = ""

    for idx, row in tqdm(questions.iterrows()):
        context = websearch_agent.invoke({"query": row["summary"]})["context"]
        if context:
            context = summary_agent.invoke(
                {"question": row["question"], "document": context}
            )
        questions.at[idx, "context"] = context
        questions.to_csv("data/questions_with_context.csv")
