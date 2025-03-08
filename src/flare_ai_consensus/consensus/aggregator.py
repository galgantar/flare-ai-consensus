import asyncio

import numpy as np

from flare_ai_consensus.router import (
    AsyncOpenRouterProvider,
    ChatRequest,
    OpenRouterProvider,
    EmbeddingRequest
)
from flare_ai_consensus.settings import AggregatorConfig, Message
from flare_ai_consensus.embeddings import EmbeddingModel

def _concatenate_aggregator(responses: dict[str, str]) -> str:
    """
    Aggregate responses by concatenating each model's answer with a label.

    :param responses: A dictionary mapping model IDs to their response texts.
    :return: A single aggregated string.
    """
    return "\n\n".join([f"{model}: {text}" for model, text in responses.items()])


def centralized_llm_aggregator(
    provider: OpenRouterProvider,
    aggregator_config: AggregatorConfig,
    aggregated_responses: dict[str, str],
) -> str:
    """Use a centralized LLM  to combine responses.

    :param provider: An OpenRouterProvider instance.
    :param aggregator_config: An instance of AggregatorConfig.
    :param aggregated_responses: A string containing aggregated
        responses from individual models.
    :return: The aggregator's combined response.
    """
    # Build the message list.
    messages: list[Message] = []
    messages.extend(aggregator_config.context)

    # Add a system message with the aggregated responses.
    aggregated_str = _concatenate_aggregator(aggregated_responses)
    messages.append(
        {"role": "system", "content": f"Aggregated responses:\n{aggregated_str}"}
    )

    # Add the aggregator prompt
    messages.extend(aggregator_config.prompt)

    payload: ChatRequest = {
        "model": aggregator_config.model.model_id,
        "messages": messages,
        "max_tokens": aggregator_config.model.max_tokens,
        "temperature": aggregator_config.model.temperature,
    }

    # Get aggregated response from the centralized LLM
    response = provider.send_chat_completion(payload)
    return response.get("choices", [])[0].get("message", {}).get("content", "")


async def async_centralized_llm_aggregator(
    provider: AsyncOpenRouterProvider,
    aggregator_config: AggregatorConfig,
    aggregated_responses: dict[str, str],
) -> str:
    """
    Use a centralized LLM (via an async provider) to combine responses.

    :param provider: An asynchronous OpenRouterProvider.
    :param aggregator_config: An instance of AggregatorConfig.
    :param aggregated_responses: A string containing aggregated
        responses from individual models.
    :return: The aggregator's combined response as a string.
    """
    messages = []
    messages.extend(aggregator_config.context)
    messages.append(
        {"role": "system", "content": f"Aggregated responses:\n{aggregated_responses}"}
    )
    messages.extend(aggregator_config.prompt)

    payload: ChatRequest = {
        "model": aggregator_config.model.model_id,
        "messages": messages,
        "max_tokens": aggregator_config.model.max_tokens,
        "temperature": aggregator_config.model.temperature,
    }

    response = await provider.send_chat_completion(payload)
    return response.get("choices", [])[0].get("message", {}).get("content", "")


async def async_centralized_embedding_aggregator(
    embedding_model: EmbeddingModel,
    responses: dict[str, str],
) -> str:
    if not responses:
        return ""

    if len(responses) == 1:
        # If there's only one response, return it directly
        return list(responses.values())[0]

    # Get embeddings for each response
    embeddings_dict = await _get_embeddings_for_responses(embedding_model, responses)
    # Calculate center of gravity (mean embedding)
    all_embeddings = np.array(list(embeddings_dict.values()))
    center_embedding = np.mean(all_embeddings, axis=0)

    # Find the response closest to the center
    closest_model_id = _find_closest_embedding(embeddings_dict, center_embedding)
    return responses[closest_model_id]


async def _get_embeddings_for_responses(
    embedding_model: EmbeddingModel, responses: dict[str, str]
) -> dict[str, np.ndarray]:
    """
    Get embeddings for each response using the provider's embedding API.

    :param provider: An asynchronous OpenRouterProvider.
    :param responses: A dictionary mapping model IDs to their response texts.
    :return: A dictionary mapping model IDs to their embedding vectors.
    """
    embeddings = {}
    for model_id, text in responses.items():
        text_ = await embedding_model.get_embeddings(text)
        embeddings[model_id] = np.array(text_.embeddings[0].values)
    return embeddings


def _find_closest_embedding(
    embeddings: dict[str, np.ndarray], center: np.ndarray
) -> str:
    """
    Find the model ID whose embedding is closest to the center.

    :param embeddings: A dictionary mapping model IDs to their embedding vectors.
    :param center: The center embedding vector.
    :return: The model ID with the closest embedding to the center.
    """
    closest_model = None
    min_distance = float("inf")

    for model_id, embedding in embeddings.items():
        # Calculate cosine similarity (or use Euclidean distance)
        distance = np.linalg.norm(embedding - center)

        if distance < min_distance:
            min_distance = distance
            closest_model = model_id

    return closest_model
