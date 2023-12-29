from dotenv import load_dotenv

load_dotenv("../.env")
import os

from typing import List, Optional, Literal
from pydantic import BaseModel, ValidationError
from uuid import uuid4
from openai import OpenAI
from .models.question_node import QuestionNode

from neomodel import db
from neo4j import GraphDatabase

username = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]
uri = os.environ["NEO4J_URI"]
driver = GraphDatabase().driver(uri, auth=(username, password))


class Question(BaseModel):
    id: int
    uuid: str | None = None
    parent_id: int | None = None
    question: str
    answer: str | None = None
    documents: List | None = None
    status: Literal["current", "unanswered", "answered"] = "unanswered"
    embedding: List[int] | None = None


class AgentRunModel:
    uuid: str
    _goal_question_id = 0
    _questions: List[Question]
    _answerpad: List[str]
    _status: Literal["init", "running", "finished"]
    _current_depth = 0

    def __init__(self):
        self._questions = []
        self._answerpad = []
        self._status = "init"
        self._current_depth = 0
        self._client = OpenAI()
        self.uuid = self.generate_uuid()

    def generate_uuid(self):
        return str(uuid4())

    def get_goal_question_id(self):
        return self._goal_question_id

    def set_current_depth(self, depth: int):
        self._current_depth = depth

    def get_current_depth(self) -> int:
        return self._current_depth

    def find_question(self, id: int) -> Question | None:
        qs = [q for q in self._questions if q.id == id]
        return None if len(qs) == 0 else qs[0]

    def set_goal(self, goal: str) -> Question:
        self.add_question({"id": self._goal_question_id, "question": goal})

    def goal(self) -> str:
        q = self.find_question(0).question
        return q if q else None

    def set_running(self):
        self._status = "running"

    def set_finished(self):
        self._status = "finished"

    def set_current_question(self, question_id: int):
        q = self.find_question(question_id)
        q.status = "current"
        return q

    def get_last_id(self) -> int:
        last_id = max([q.id for q in self._questions])
        return last_id

    def get_all_questions(self) -> List[Question]:
        return self._questions

    def get_answered_questions(self) -> List[Question]:
        qs = [q for q in self._questions if q.status == "answered"]
        return qs

    def get_unanswered_questions(self) -> List[Question]:
        qs = [q for q in self._questions if q.status == "unanswered"]
        return qs

    def get_current_question(self) -> Question:
        qs = [q for q in self._questions if q.status == "current"]
        return None if len(qs) == 0 else qs[0]

    def add_question(self, q_dict: dict, add_embeddings=False):
        try:
            q = Question(**q_dict)
            q.uuid = self.generate_uuid()
            if add_embeddings:
                self.add_embedding_to_question(q)
            self._questions.append(q)
        except ValidationError as e:
            print(e.errors())

    def add_questions(self, q_list: list, add_embeddings=False):
        for q in q_list:
            self.add_question(q, add_embeddings)

    def add_answer_to_question(self, question_id: int, answer: str, documents=None):
        q = self.find_question(question_id)
        if not q:
            return False
        q.answer, q.documents = answer, documents
        q.status = "answered"
        return True

    def add_answer_to_answerpad(self, answer: str):
        self._answerpad.append(answer)

    def get_answerpad(self) -> List[str]:
        return self._answerpad

    def add_embedding_to_question(self, question: Question):
        response = self._client.embeddings.create(
            input=question.question, model="text-embedding-ada-002"
        )
        question.embedding = response.data[0].embedding

    def create_tree(self, id=None, parent_id=None):
        "Creates a tree structure with the questions"
        id = id if id else self._goal_question_id
        question = next((q for q in self._questions if q.id == id), None)
        tree = {
            "id": id,
            "edge_name": question.question,
            "name": question.question,
            "answer": question.answer,
            "parent": parent_id,
        }
        children_ids = [q.id for q in self._questions if q.parent_id == id]

        if len(children_ids) != 0:
            tree["children"] = []
            for cid in children_ids:
                child_tree = self.create_tree(cid, id)
                tree["children"].append(child_tree)
        return tree

    def save(self):
        q_nodes = {}
        for q in self._questions:
            params = (
                q.model_dump(include={"uid", "question", "answer", "embedding"})
                | {"run_id": self.uuid}
                | {"goal": (q.id == 0)}
            )
            q_nodes[q.id] = QuestionNode(**params)

        self.save_to_db(q_nodes)
        return self.uuid

    def save_to_db(self, nodes):
        db.set_connection(driver=driver)
        with db.transaction:
            for q in self._questions:
                nodes[q.id].save()

            for q in self._questions:
                if q.parent_id == None:
                    continue
                nodes[q.parent_id].follow_up_question.connect(nodes[q.id])
        db.close_connection()
