from .prompts.most_pertinent_question import mostPertinentQuestion
from .prompts.create_questions import createQuestions
from .prompts.retrieval_qa import retrievalQA
from .prompts.refine_answer import refineAnswer
from .prompts.hypothetical_answer import hyDE
from .prompts.compile_answer import compileAnswer
from .helpers.response_helpers import question_str_to_dict
from .helpers.response_helpers import result_to_questions_list
from .agent_run_model import AgentRunModel, Question
from yachalk import chalk



def select_next_question(run_model: AgentRunModel, agent_settings) -> Question:
    print(chalk.bold.green(f"\n\n╰─➤ [{run_model.get_current_depth()}] Next Question I must ask ▷▶\n"))
    unanswered_questions = run_model.get_unanswered_questions()
    q_list = [f"{q.id}. '{q.question}'" for q in unanswered_questions]
    q_string_list = "[\n" + ",\n".join(q_list) + "\n]"
    retry = 0
    should_retry = True
    while retry < 3 and should_retry:
        try:
            pq_res = mostPertinentQuestion(
                run_model.goal(),
                q_string_list,
                model=agent_settings.get("model", "mistral-openorca:latest")
            )
            pq = question_str_to_dict(pq_res)
            should_retry = False
        except:
            retry += 1
            print("\n###\nCould not parse the next questions \nRetrying...\n###\n")

    next_question = run_model.find_question(pq["id"])
    return next_question


def ask_new_questions(run_model: AgentRunModel, agent_settings, parent_node_id):
    print(chalk.bold.green(f"\n[{run_model.get_current_depth()}]\n╰─➤ Ask more questions based on new context ▷▶\n"))
    start_id = run_model.get_last_id() + 1
    q_list = [f"'{q.question}'" for q in run_model.get_all_questions()]
    previous_questions = "[\n" + ",\n".join(q_list) + "\n]"
    retry = 0
    should_retry = True
    while retry < 3 and should_retry:
        try:
            questions_res = createQuestions(
                run_model.goal(),
                run_model.get_answerpad()[-1],
                previous_questions,
                agent_settings.get("num_questions_per_iter", 3),
                model=agent_settings.get("model", "mistral-openorca:latest"),
            )
            questions = result_to_questions_list(
                questions_res, start_id, parent_node_id
            )
            should_retry = False
        except:
            retry += 1
            print("\n###\nCould not parse question list \nRetrying...\n###\n")

    return questions


def answer_current_question(run_model: AgentRunModel, agent_settings, docs):
    print(chalk.bold.green(f"\n[{run_model.get_current_depth()}]\n╰─➤ Answer ▷▶\n"))
    current_question = run_model.get_current_question()
    docs_string = "\n----\n".join(
        [f"\nSource Text: {doc.page_content}\nSource Metadata: {doc.metadata}\n" for doc in docs]
    )
    print(chalk.gray([d.metadata for d in docs]))
    print(chalk.gray(docs_string[:250]))
    print(chalk.gray(current_question.question))
    intermediate_q_answer = retrievalQA(
        current_question.question,
        docs_string,
        model=agent_settings.get("model", "mistral-openorca:latest"),
    )
    run_model.add_answer_to_question(current_question.id, intermediate_q_answer, docs)

    return intermediate_q_answer


def refine_goal_answer(run_model: AgentRunModel, agent_settings, new_context: str):
    print(chalk.bold.green(f"\n[{run_model.get_current_depth()}]\n╰─➤ Refining the existing answer ▷▶\n"))
    goal_question_id = run_model.get_goal_question_id()
    refine_answer_res = refineAnswer(
        question=run_model.goal(),
        answer=run_model.find_question(goal_question_id).answer,
        context=new_context,
        model=agent_settings.get("model", "mistral-openorca:latest"),
    )
    run_model.add_answer_to_answerpad(refine_answer_res)
    run_model.add_answer_to_question(goal_question_id, refine_answer_res)


def create_initial_hypothesis(run_model: AgentRunModel, agent_settings):
    print(chalk.bold.green(f"\n╰─➤ Initial Hypothesis ▷▶\n"))
    hyde_res = hyDE(
        run_model.goal(), 
        model=agent_settings.get("model", "mistral-openorca:latest")
    )
    run_model.add_answer_to_answerpad(hyde_res)
    return hyde_res


def compile_answer(run_model: AgentRunModel, agent_settings):
    print(
        chalk.bold.green(
            f"\n[{run_model.get_current_depth()}]\n╰─➤ Compiling the final Answer ▷▶\n"
        )
    )
    notes = "\n...".join([q.answer for q in run_model.get_answered_questions()])
    answer_res = compileAnswer(
        question=run_model.goal(),
        context=notes,
        model=agent_settings.get("model", "mistral-openorca:latest"),
    )
    run_model.add_answer_to_answerpad(answer_res)
    goal_question_id = run_model.get_goal_question_id()
    run_model.add_answer_to_question(goal_question_id, answer_res)
