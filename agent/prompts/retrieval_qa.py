from ..ollama import client

from .prompt_logger import log_prompt


def retrievalQA(
    question: str,
    documents: str,
    model="mistral-openorca:latest",
    stream=True,
    verbose=False,
):
    SYS_PROMPT = (
        "You will be provided with a 'question' and a list of 'excerpts' from a long document (delimited by ```)."
        " Your task is to answer the question based on the given documents excerpts."
        " Answer in a crisp, objective and business casual tone."
    )

    prompt = (
        f"Question:\n ``` {question} ```\n"
        f"Excerpts:\n ``` {documents} ```\n"
        "Your response:"
    )

    if verbose:
        log_prompt("\n---\Retrieval QA Prompt\n")
        log_prompt(f"SYS_PROMPT: {SYS_PROMPT}\n")
        log_prompt(f"QUESTIONS: {question}\n")
        log_prompt(f"DOCS: {documents[:300]}\n...\n{documents[-300:]}\n---\n")

    response, _ = client.generate(
        model_name=model, system=SYS_PROMPT, prompt=prompt, stream=stream
    )

    return response
