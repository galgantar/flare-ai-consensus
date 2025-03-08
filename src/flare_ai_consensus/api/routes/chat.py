import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import json

from flare_ai_consensus.consensus import run_consensus
from flare_ai_consensus.router import AsyncOpenRouterProvider
from flare_ai_consensus.settings import ConsensusConfig, Message
from flare_ai_consensus.embeddings import EmbeddingModel

logger = structlog.get_logger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    """
    Pydantic model for chat message validation.

    Attributes:
        message (str): The chat message content, must not be empty
    """

    system_message: str = Field(..., min_length=1)
    user_message: str = Field(..., min_length=1)


class ChatRouter:
    """
    A simple chat router that processes incoming messages using the CL pipeline.
    """

    def __init__(
            self,
            router: APIRouter,
            provider: AsyncOpenRouterProvider,
            embedding_model: EmbeddingModel,
            consensus_config: ConsensusConfig | None = None,
    ) -> None:
        """
        Initialize the ChatRouter.

        Args:
            router (APIRouter): FastAPI router to attach endpoints.
            provider: instance of an async OpenRouter client.
            consensus_config: config for running the consensus algorithm.
        """
        self._router = router
        self.provider = provider
        self.embedding_model = embedding_model
        if consensus_config:
            self.consensus_config = consensus_config
        self.logger = logger.bind(router="chat")
        self._setup_routes()

    def _setup_routes(self) -> None:
        """
        Set up FastAPI routes for the chat endpoint.
        """

        @self._router.post("/")
        async def chat(message: ChatMessage) -> dict[str, str] | None:  # pyright: ignore [reportUnusedFunction]
            """
            Process a chat message through the CL pipeline.
            Returns an aggregated response after a number of iterations.
            """
            try:
                self.logger.debug("Received chat message", message=message.user_message)
                # Build initial conversation
                initial_conversation: list[Message] = [
                    {
                        "role": "system", "content": """You are an advanced crypto trading bot specializing in SWAP and LEND operations on the following tokens: WFLR, WETH, FLR, USDX, USDT. You have access to real-time wallet balances, current exchange rates, and lending opportunities. Your goal is to execute profitable transactions while managing risk.  
                        When deciding on a transaction, you must:  
                        1. Analyze Market Conditions – Consider current exchange rates, arbitrage opportunities, and lending yields.  
                        2. Optimize for Profitability – Prioritize swaps that increase portfolio value or lending that provides the best APY.  
                        3. Risk Management – Avoid excessive exposure to volatile assets or illiquid markets.  
                        4. Provide Justification – Explain each transaction thoroughly with data-driven reasoning.  
                        
                        Your response must be in the following JSON format:  
                        
                        {"operation": "SWAP/LEND", "token_a": "TOKEN1", "token_b": "TOKEN2", "amount": "AMOUNT", "reason": "Concise explanation of why this trade is optimal based on market conditions, risk, and profit potential."}  
                        
                        It is neccessary that you create only one transaction per response.
                        
                        Example 1 (SWAP):  
                        {"operation": "SWAP", "token_a": "WFLR", "token_b": "USDX", "amount": "500", "reason": "WFLR is currently overbought, and USDX provides a stable hedge. The WFLR/USDX rate is at a local high, making this a profitable swap."}  
                        
                        Example 2 (LEND):  
                        {"operation": "LEND", "token_a": "USDT", "token_b": "USDX", "amount": "1000", "reason": "USDT lending pool is currently offering 8% APY, significantly higher than holding in USDX. Lending maximizes passive yield."}  
                        
                        Continuously monitor market conditions and execute optimal trades while minimizing risk.  
                        """
                    },
                    {"role": "user", "content": message.user_message},
                ]

                # Run consensus algorithm
                answer, shapley_values = await run_consensus(
                    self.provider,
                    self.consensus_config,
                    [initial_conversation],
                    self.embedding_model
                )

            except Exception as e:
                self.logger.exception("Chat processing failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e
            else:
                self.logger.info("Response generated", answer=answer)
                return {"response": answer, "shapley_values": json.dumps(shapley_values)}

    @property
    def router(self) -> APIRouter:
        """Return the underlying FastAPI router with registered endpoints."""
        return self._router
