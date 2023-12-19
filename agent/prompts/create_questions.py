from ..ollama import client
from .prompt_logger import log_prompt

def createQuestions(
    question: str,
    context: str,
    previous_questions: str,
    num_questions=2,
    model="mistral-openorca:latest",
    stream=True,
    verbose=False
):
    SYS_PROMPT = (
        "Your are a curious researcher. Your task is to ask questions that can help you answer the goal question. "
        "You will be given a 'goal question', a 'context' and some 'previously asked questions' as input (delimited by ```). "
        "Use the following chain of thoughts:\n"
        "Thought 1: Use only the given 'context' and the 'goal question', and no other previous knowledge."
        "Thought 2: Think about questions you can ask that can not be answered using the given context.\n"
        "Thought 3: Think about if these questions are relevant to your goal question. Discard the questions that are not relevant.\n"
        "Thought 4: Discard the questions that are semantically similar to the 'previously asked questions'.\n"
        f"Respond with at most {num_questions} questions. "
        "Format your response as an array of strings. Example: \n "
        " ['First Question', 'Second Question', ...]"
    )

    prompt = (
        f"Goal Question: ``` {question} ```.\n\n"
        f"Context: \n ``` {context} ``` \n\n"
        f"Previously Asked Questions:  ``` {previous_questions} ```\n\n"
        "Your response:"
    )

    if verbose:
        log_prompt("\n---\nCreate Questions Prompt\n")
        log_prompt(f"SYS_PROMPT: {SYS_PROMPT}\n")
        log_prompt(f"QUESTION: {question}\n")
        log_prompt(f"CTX: {context[:300]}\n...\n{context[-300:]}\n")
        log_prompt(f"PREV QUESTIONS: {previous_questions}\n---\n")

    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt, stream=stream)

    return response
