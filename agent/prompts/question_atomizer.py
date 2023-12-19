from ..ollama import client
from .prompt_logger import log_prompt


def questionAtomizer(
    question: str,
    num_questions=2,
    model="mistral-openorca:latest",
    stream=True,
    verbose=False,
):
    SYS_PROMPT = (
        "You are a curious researcher. You will be provided with a goal question. "
        "Your task is to split the goal question into smaller questions if possible. "
        "Do not use any prior knowledge. "
        "Think along the following chain of thoughts:\n"
        "Thought 1: Consider components like the Subject, Predicate, Characters, Ideas, Concepts, Entities, Actions, Or Events mentioned in the goal question.\n"
        "Thought 2: Divide the goal question into multiple small, simple and atomistic sub questions.\n"
        "Thought 3: Think if these questions semantically different from each other. Discard semantically similar questions.\n"
        f"Respond with at most {num_questions} sub questions"
        "Format your response like: \n"
        "n. First question\n"
        "n. Second question\n"
    )

    prompt = f"Goal Question: ``` {question} ```.\n\n Your response:"

    if verbose:
        log_prompt("\n---\nQuestion Atomiser Prompt\n")
        log_prompt(f"SYS_PROMPT: {SYS_PROMPT}\n")
        log_prompt(f"QUESTIONS: {question}\n")

    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt, stream=stream)

    return response
