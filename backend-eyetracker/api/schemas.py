from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional


class SourceUrl(BaseModel):
    url: str = Field(..., description="Website URL")
    title: str = Field("", description="Page title")
    domain: str = Field("", description="Website domain")


class ScrapedData(BaseModel):
    url: str = Field(..., description="Website URL")
    title: str = Field("", description="Page title")
    content: str = Field("", description="Extracted text content")
    word_count: int = Field(0, description="Number of words")
    quality_score: int = Field(0, ge=0, le=100, description="Content quality score")
    quality_tier: str = Field("UNKNOWN", description="Quality tier")
    scraping_method: str = Field("unknown", description="Scraping method")
    scraping_success: bool = Field(False, description="Whether scraping succeeded")
    domain: str = Field("", description="Website domain")
    snippet: str = Field("", description="Content snippet")
    error_message: str = Field("", description="Error message if scraping failed")
    relevance_score: int = Field(0, description="LLM relevance score")


class ScraperRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, description="Search query")
    max_results: int = Field(5, ge=1, le=15, description="Maximum number of results")
    url_multiplier: int = Field(4, ge=2, le=10, description="URL multiplier for ranking (2-10x)")

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class ScraperResponse(BaseModel):
    query: str = Field(..., description="Search query used")
    scraped_data: List[ScrapedData] = Field(..., description="Raw scraping results for each URL")
    total_urls: int = Field(..., description="Total URLs processed")
    successful_scrapes: int = Field(..., description="Number of successful scrapes")
    failed_scrapes: int = Field(..., description="Number of failed scrapes")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    url_multiplier_used: int = Field(..., description="URL multiplier used")


class SearchRequest(ScraperRequest):
    pass  # Same fields as ScraperRequest


class SearchResponse(BaseModel):
    query: str = Field(..., description="Original search query")
    source_urls: List[SourceUrl] = Field(..., description="List of source URLs used")
    key_points: List[str] = Field(..., description="LLM-extracted key points")
    summary: str = Field("", description="LLM-generated summary")
    unified_content: str = Field("", description="LLM-organized unified content")
    total_sources: int = Field(..., description="Number of sources used")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    url_multiplier_used: int = Field(..., description="URL multiplier used")
    enhanced_query: str = Field(..., description="LLM-enhanced query used for optimization")
    query_enhancement_applied: bool = Field(..., description="Whether query enhancement was applied")


class OptimizerResponse(BaseModel):
    query: str = Field(..., description="Original search query")
    optimized_data: List[ScrapedData] = Field(..., description="Vector-optimized scraped data")
    total_original_sources: int = Field(..., description="Number of original sources")
    total_optimized_sources: int = Field(..., description="Number of optimized sources")
    optimization_stats: Dict[str, Any] = Field(..., description="Optimization statistics")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    url_multiplier_used: int = Field(..., description="URL multiplier used")


class AudioTranscriptionResponse(BaseModel):
    text: str = Field(..., description="Transcribed text from audio")

class DuckyScriptResponse(BaseModel):
    output_ducky_script: str = Field(..., description="Rubber Ducky script for computer control")


class BotCommand(BaseModel):
    L: int = Field(0, ge=-100, le=100, description="Left wheel speed (-100 to 100)")
    R: int = Field(0, ge=-100, le=100, description="Right wheel speed (-100 to 100)")
    S1: int = Field(90, ge=20, le=160, description="Head tilt angle (20 to 160)")
    S2: int = Field(90, ge=0, le=110, description="Head lift angle (0 to 110)")


class BotCommandResponse(BaseModel):
    success: bool = Field(..., description="Whether command was executed")
    message: str = Field(..., description="Status message")
    command_sent: Dict[str, Any] = Field(default_factory=dict, description="Actual command sent to bot")
    values_clamped: Optional[Dict[str, Any]] = Field(None, description="Original vs clamped values")


class ConversationEntry(BaseModel):
    timestamp: str = Field(..., description="ISO format timestamp")
    session_id: str = Field(..., description="Session identifier")
    user_input: str = Field(..., description="User's input query")
    bot_response: str = Field(..., description="Bot's response")
    subsystems: Dict[str, Any] = Field(default_factory=dict, description="Output from each subsystem")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ManagerResponse(BaseModel):
    status: str = Field(..., description="Response status (success/error)")
    user_query: str = Field(..., description="Original user query")
    classification: Dict[str, Any] = Field(..., description="Query classification result")
    subsystems_activated: List[str] = Field(..., description="List of activated subsystems")
    subsystems_outputs: Dict[str, Any] = Field(..., description="Output from each subsystem")
    final_response: str = Field(..., description="Final response to user")
    conversation_history: List[ConversationEntry] = Field(
        ..., 
        description="Recent conversation history (last 5)"
    )