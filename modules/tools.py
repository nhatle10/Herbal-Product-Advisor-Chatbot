from pydantic import BaseModel, Field

class Retrieve(BaseModel):
    """
    Searches the knowledge base for answers. The query parameter should contain the contextualized question to search for in the knowledge base.
    """
    query: str = Field(description="should be a search query")

class RefuseToAnswer(BaseModel):
    """
    Model to indicate refusal to answer and guide the conversation back.
    """
    reason: str = Field(description="Reason for refusing to answer.")

class ContactShop(BaseModel):
    """
    Provides the shop's contact information to the user.
    """
    pass