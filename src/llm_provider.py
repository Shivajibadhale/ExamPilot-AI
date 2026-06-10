from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def build_chat_model(
    provider: str,
    model_name: str,
    api_key: str,
    temperature: float = 0.25,
) -> BaseChatModel:
    if provider == "Gemini":
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
        )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
    )

def build_embedding_model(provider: str, api_key: str) -> Embeddings:
    if provider == "Gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
            task_type="RETRIEVAL_DOCUMENT",
        )

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
