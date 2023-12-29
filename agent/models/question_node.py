from neomodel import (
    StructuredNode,
    StringProperty,
    UniqueIdProperty,
    RelationshipTo,
    ArrayProperty,
    FloatProperty,
    Relationship,
    IntegerProperty,
    StructuredRel,
    BooleanProperty,
)



class QuestionNode(StructuredNode):
    goal = BooleanProperty()
    uid = StringProperty()
    run_id = StringProperty()
    question = StringProperty()
    answer = StringProperty()
    embedding = ArrayProperty(FloatProperty())

    ## Relationships
    follow_up_question = RelationshipTo("QuestionNode", relation_type="FOLLOW_UP_QUESTION")
    semantic_similarity = Relationship("QuestionNode", relation_type="SEMANTIC_SIMILARITY")



