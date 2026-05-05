# generated-by: codex-agent 2025-02-15T00:16:00Z
"""
RAG domain models.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .common import ObjectIdStr, RagSlug, UUIDStr

DocStatusEnum = Literal["uploaded", "chunking", "chunked", "indexing", "indexed", "error"]
IndexStatusEnum = Literal["idle", "building", "error"]
VisibilityEnum = Literal["private", "public"]
RAG_SYSTEM_PROMPT_MAX_LENGTH = 4000
DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are a helpful and warm assistant. Use ONLY the provided context. "
    "Every claim must be supported by the context; do not add generic advice or steps that are not present in the excerpts. "
    "Do not add external or prior knowledge. If the context is thin, give a short, cautious answer and state that more detail is not available in the excerpts. "
    "If the answer is long, provide a structured synthesis in 5 to 8 points maximum (with sub-points if needed). "
    "Respond in the user's language using GitHub-flavored Markdown."
)


def normalize_rag_system_prompt(value: Optional[str]) -> str:
    text = (value or "").strip()
    return text or DEFAULT_RAG_SYSTEM_PROMPT


class RagDoc(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat().replace("+00:00", "Z")})

    doc_id: UUIDStr
    filename: str
    path: Optional[str] = None
    mime: Literal[
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.presentation",
        "text/plain",
        "application/msword",
    ]
    size_bytes: int = Field(ge=0)
    sha256: Optional[str] = None
    status: DocStatusEnum
    chunk_count: int = Field(ge=0)
    error: Optional[str] = None
    uploaded_at: datetime
    indexed_at: Optional[datetime] = None

class RagSummary(BaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda dt: dt.isoformat().replace("+00:00", "Z")})

    slug: RagSlug
    name: str
    description: str
    chunks: int = Field(ge=0)
    last_updated: datetime
    visibility: VisibilityEnum = Field(default="private")
    suggestions: List[str] = Field(default_factory=list)
    suggestions_en: List[str] = Field(default_factory=list)
    suggestions_lang: Optional[str] = Field(default=None, description="BCP-47 language code for primary suggestions")


class RagAdminDetail(RagSummary):
    system_prompt: str = Field(
        default=DEFAULT_RAG_SYSTEM_PROMPT,
        max_length=RAG_SYSTEM_PROMPT_MAX_LENGTH,
    )


class RagListResponse(BaseModel):
    items: List[RagSummary]
    next_cursor: Optional[str] = None
    corr_id: str


class RagAdminDetailResponse(BaseModel):
    rag: RagAdminDetail
    corr_id: str


class RagDocsResponse(BaseModel):
    rag_slug: RagSlug
    docs: List[RagDoc]
    corr_id: str


class IndexStatusResponse(BaseModel):
    rag_slug: RagSlug
    status: IndexStatusEnum
    progress: float = Field(ge=0, le=1)
    total_docs: int = Field(ge=0)
    done_docs: int = Field(ge=0)
    total_chunks: int = Field(ge=0)
    done_chunks: int = Field(ge=0)
    error: Optional[str] = None
    corr_id: str


class RagUploadAccepted(BaseModel):
    job_id: UUIDStr
    accepted: List[str]
    skipped: List[str]
    rag_slug: RagSlug
    corr_id: str


class RagCreateRequest(BaseModel):
    slug: RagSlug
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    visibility: VisibilityEnum = Field(default="private")
    system_prompt: Optional[str] = Field(default=None, max_length=RAG_SYSTEM_PROMPT_MAX_LENGTH)

    @field_validator("system_prompt")
    @classmethod
    def trim_system_prompt(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value is not None else None


class RagCreateResponse(BaseModel):
    rag: RagAdminDetail
    corr_id: str


class RagUpdateRequest(BaseModel):
    rag_slug: RagSlug
    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=500)
    visibility: Optional[VisibilityEnum] = None
    system_prompt: Optional[str] = Field(default=None, max_length=RAG_SYSTEM_PROMPT_MAX_LENGTH)

    @field_validator("system_prompt")
    @classmethod
    def trim_system_prompt(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def validate_has_update(self) -> "RagUpdateRequest":
        if (
            self.name is None
            and self.description is None
            and self.visibility is None
            and self.system_prompt is None
        ):
            raise ValueError("Provide at least one RAG metadata field to update")
        return self


class RagUpdateResponse(BaseModel):
    rag: RagAdminDetail
    corr_id: str


class RagDeleteRequest(BaseModel):
    rag_slug: RagSlug
    confirmation: RagSlug


class RagDeleteResponse(BaseModel):
    deleted: Literal[True]
    rag_slug: RagSlug
    corr_id: str


class RagUser(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    id: ObjectIdStr = Field(alias="_id")
    email: str
    name: str
    role: Literal["user", "admin"]


class RagUsersResponse(BaseModel):
    rag_slug: RagSlug
    users: List[RagUser]
    corr_id: str


class RagUserUpsertRequest(BaseModel):
    rag_slug: RagSlug
    user_id: Optional[ObjectIdStr] = None
    email: Optional[str] = None

    @model_validator(mode="before")
    def validate_one_of(
        cls,
        data: dict,
    ) -> dict:
        user_id = data.get("user_id")
        email = data.get("email")
        has_id = user_id is not None and user_id != ""
        has_email = email is not None and email != ""
        if has_id == has_email:
            raise ValueError("Provide exactly one of user_id or email")
        return data


class RagUserUpsertResponse(BaseModel):
    rag_slug: RagSlug
    user_id: ObjectIdStr
    corr_id: str
