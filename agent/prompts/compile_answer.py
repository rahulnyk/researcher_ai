from ..ollama import client
from .prompt_logger import prompt_logger

def compileAnswer(
    question: str,
    context: str,
    model="mistral-openorca:latest",
    stream=True,
    verbose=False
):
    SYS_PROMPT = (
        "You are a Research Assistant.\n"
        "You will be provided with a question, and some context."
        " Your task is to answer the question using the given context and no other previous knowledge."
        " If the provided research notes are not relevant to the question, respond with 'Can not answer the question with given context'"
        " Write an Elaborate and detailed answer, itomize whenever possible, use business casual language."
    )

    prompt = (
        f"Question: ``` {question} ```\n"
        f"Context: ``` {context} ```\n"
        "Your Answer:"
    )

    if verbose:
        prompt_logger.critical("\n---\nCompile Answer Prompt:\n")
        prompt_logger.critical(f"SYS_PROMPT: {SYS_PROMPT}\n")
        prompt_logger.critical(
            f"QUESTIONS: {question} \nDOCS: {context[:500]}\n...\n{context[-500:]} \n---\n "
        )

    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt, stream=stream)

    return response
