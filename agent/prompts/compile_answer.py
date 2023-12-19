from ..ollama import client
from yachalk import chalk
from .prompt_logger import log_prompt

def compileAnswer(
    question: str,
    context: str,
    model="mistral-openorca:latest",
    stream=True,
    verbose=False
):
    SYS_PROMPT = (
        "You will be provided with a question, and some research notes."
        " Your task is to answer the question using the research notes."
        " Use only the given research notes and no other pre-existing knowledge to write your answer.\n"
        " Use an elaborate and descriptive style, itomize whenever possible, use business casual language."
        " If the provided research notes are not relevant to the question, respond with  'Can not answer the question with given context'"
    )

    prompt = (
        f"Question: ``` {question} ```\n"
        f"Research Notes: ``` {context} ```\n"
        "Your Answer:"
    )

    if verbose:
        log_prompt("\n---\nCompile Answer Prompt:\n")
        log_prompt(f"SYS_PROMPT: {SYS_PROMPT}\n")
        log_prompt(
            f"QUESTIONS: {question} \nDOCS: {context[:500]}\n...\n{context[-500:]} \n---\n "
        )

    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt, stream=stream)

    return response
