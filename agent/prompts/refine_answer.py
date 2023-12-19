from ..ollama import client
from .prompt_logger import log_prompt


def refineAnswer(
    question: str,
    answer: str,
    context: str,
    model="mistral-openorca:latest",
    stream=True,
    verbose=False,
):
    SYS_PROMPT = (
        "You will be provided with a question, an existing answer, and some new context"
        " You have the opportunity to improve upon the existing answer using the new context."
        " Thinking about the following:\n"
        "\t - What is the information in the new context that is relevant to the question?.\n"
        "\t - Can you augment or correct the existing answer using the new context?\n"
        " Use only the given context and the existing answer and no other pre-existing knowledge.\n"
        " Use a elaborate and descriptive style, itomize whenever possible, use business casual language."
    )

    prompt = (
        f"Question: ``` {question} ```\n"
        f"Existing Answer: ``` {answer} ```\n"
        f"New Context: ``` {context} ```\n"
        "New Answer:"
    )

    if verbose:
        log_prompt("\n---\nRefine Answer Prompt\n")
        log_prompt(f"SYS_PROMPT: {SYS_PROMPT}\n")
        log_prompt(f"QUESTIONS: {question}\nANSWER: {answer}\n")
        log_prompt(f"CTX: {context[:300]}\n...\n{context[-300:]}\n---\n")

    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt, stream=stream)

    return response
