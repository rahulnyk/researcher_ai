from ..ollama import client

from .prompt_logger import prompt_logger


def retrievalQA(
    question: str,
    documents: str,
    model="mistral-openorca:latest",
    stream=True,
    verbose=False,
):
    SYS_PROMPT = (
        "You will be provided with a Question and some context (delimited by ```)."
        " Your task is to answer the given question based on the given context."
        " Write your answer using only the context and no other previous knowledge. "
        " Answer in a crisp, objective and business casual tone."
    )

    prompt = (
        f"Question: ``` {question} ```\n"
        f"Context:\n ``` {documents} ```\n"
        "Your response:"
    )

    if verbose:
        prompt_logger.critical("\n---\nRetrieval QA Prompt\n")
        prompt_logger.critical(f"SYS_PROMPT: {SYS_PROMPT}\n")
        prompt_logger.critical(f"QUESTIONS: {question}\n")
        prompt_logger.critical(f"DOCS: {documents[:300]}\n...\n{documents[-300:]}\n---\n")

    response, _ = client.generate(
        model_name=model, system=SYS_PROMPT, prompt=prompt, stream=stream
    )

    return response
