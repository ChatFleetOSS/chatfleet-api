// generated-by: codex-agent 2025-02-15T00:45:00Z
import { z } from "zod";

/** ------------------ primitives ------------------ */
export const ObjectId = z
  .string()
  .regex(/^[a-fA-F0-9]{24}$/, "Invalid ObjectId");
export const UUID = z.string().uuid();
export const ISODate = z.string().datetime();
export const Email = z.string().email();
export const RagSlug = z
  .string()
  .min(1)
  .max(80)
  .regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, "slug must be lowercase URL-safe");

export const Role = z.enum(["user", "admin"]);

/** ------------------ errors ------------------ */
export const ErrorEnvelope = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
  }),
  corr_id: UUID,
});
export type ErrorEnvelope = z.infer<typeof ErrorEnvelope>;

/** ------------------ users/auth ------------------ */
export const UserPublic = z.object({
  _id: ObjectId,
  email: Email,
  name: z.string(),
  role: Role,
  rags: z.array(RagSlug),
  created_at: ISODate,
  updated_at: ISODate,
});
export type UserPublic = z.infer<typeof UserPublic>;

export const RegisterRequest = z.object({
  email: Email,
  name: z.string().min(1),
  password: z.string().min(8),
});
export type RegisterRequest = z.infer<typeof RegisterRequest>;

export const LoginRequest = z.object({
  email: Email,
  password: z.string().min(8),
});
export type LoginRequest = z.infer<typeof LoginRequest>;

export const AuthResponse = z.object({
  token: z.string(),
  user: UserPublic,
  corr_id: UUID,
});
export type AuthResponse = z.infer<typeof AuthResponse>;

export const UsersListResponse = z.object({
  items: z.array(UserPublic),
  next_cursor: z.string().nullable().optional(),
  corr_id: UUID,
});
export type UsersListResponse = z.infer<typeof UsersListResponse>;

export const AdminCreateUserRequest = z.object({
  email: Email,
  name: z.string(),
  role: Role,
  rags: z.array(RagSlug).optional(),
});
export type AdminCreateUserRequest = z.infer<
  typeof AdminCreateUserRequest
>;

/** ------------------ RAGs ------------------ */
export const RagSummary = z.object({
  slug: RagSlug,
  name: z.string(),
  description: z.string(),
  chunks: z.number().int().min(0),
  last_updated: ISODate,
});
export type RagSummary = z.infer<typeof RagSummary>;

export const RagListResponse = z.object({
  items: z.array(RagSummary),
  next_cursor: z.string().nullable().optional(),
  corr_id: UUID,
});
export type RagListResponse = z.infer<typeof RagListResponse>;

export const DocStatusEnum = z.enum([
  "uploaded",
  "chunking",
  "chunked",
  "indexing",
  "indexed",
  "error",
]);

export const RagDoc = z.object({
  doc_id: UUID,
  filename: z.string(),
  path: z.string().nullable().optional(),
  mime: z.enum(["application/pdf"]),
  size_bytes: z.number().int().min(0),
  sha256: z.string().nullable().optional(),
  status: DocStatusEnum,
  chunk_count: z.number().int().min(0),
  error: z.string().nullable().optional(),
  uploaded_at: ISODate,
  indexed_at: ISODate.nullable().optional(),
});
export type RagDoc = z.infer<typeof RagDoc>;

export const RagDocsResponse = z.object({
  rag_slug: RagSlug,
  docs: z.array(RagDoc),
  corr_id: UUID,
});
export type RagDocsResponse = z.infer<typeof RagDocsResponse>;

export const RagUploadAccepted = z.object({
  job_id: UUID,
  accepted: z.array(z.string()),
  skipped: z.array(z.string()),
  rag_slug: RagSlug,
  corr_id: UUID,
});
export type RagUploadAccepted = z.infer<typeof RagUploadAccepted>;

export const RagCreateRequest = z.object({
  slug: RagSlug,
  name: z.string().min(1).max(120),
  description: z.string().max(500).default(""),
});
export type RagCreateRequest = z.infer<typeof RagCreateRequest>;

export const RagCreateResponse = z.object({
  rag: RagSummary,
  corr_id: UUID,
});
export type RagCreateResponse = z.infer<typeof RagCreateResponse>;

export const RagDeleteRequest = z.object({
  rag_slug: RagSlug,
  confirmation: RagSlug,
});
export type RagDeleteRequest = z.infer<typeof RagDeleteRequest>;

export const RagDeleteResponse = z.object({
  deleted: z.literal(true),
  rag_slug: RagSlug,
  corr_id: UUID,
});
export type RagDeleteResponse = z.infer<typeof RagDeleteResponse>;

export const IndexStatusEnum = z.enum(["idle", "building", "error"]);
export const IndexStatusResponse = z.object({
  rag_slug: RagSlug,
  status: IndexStatusEnum,
  progress: z.number().min(0).max(1),
  total_docs: z.number().int().min(0),
  done_docs: z.number().int().min(0),
  total_chunks: z.number().int().min(0),
  done_chunks: z.number().int().min(0),
  error: z.string().nullable().optional(),
  corr_id: UUID,
});
export type IndexStatusResponse = z.infer<typeof IndexStatusResponse>;

/** ------------------ access control ------------------ */
export const RagUsersResponse = z.object({
  rag_slug: RagSlug,
  users: z.array(
    z.object({
      _id: ObjectId,
      email: Email,
      name: z.string(),
      role: Role,
    })
  ),
  corr_id: UUID,
});
export type RagUsersResponse = z.infer<typeof RagUsersResponse>;

export const RagUserUpsertRequest = z
  .object({
    rag_slug: RagSlug,
    user_id: ObjectId.optional(),
    email: Email.optional(),
  })
  .superRefine((val, ctx) => {
    const hasId = !!val.user_id;
    const hasEmail = !!val.email;
    if (hasId === hasEmail) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "Provide exactly one of user_id or email",
        path: ["user_id", "email"],
      });
    }
  });
export type RagUserUpsertRequest = z.infer<typeof RagUserUpsertRequest>;

export const RagUserUpsertResponse = z.object({
  rag_slug: RagSlug,
  user_id: ObjectId,
  corr_id: UUID,
});
export type RagUserUpsertResponse = z.infer<typeof RagUserUpsertResponse>;

/** ------------------ chat ------------------ */
export const MessageRole = z.enum(["system", "user", "assistant"]);
export const ChatMessage = z.object({
  role: MessageRole,
  content: z.string(),
});
export type ChatMessage = z.infer<typeof ChatMessage>;

export const ChatOptions = z
  .object({
    top_k: z.number().int().min(1).default(6),
    temperature: z.number().min(0).max(1).default(0.2),
    max_tokens: z.number().int().min(1).default(500),
  })
  .partial();
export type ChatOptions = z.infer<typeof ChatOptions>;

export const ChatRequest = z.object({
  rag_slug: RagSlug,
  messages: z.array(ChatMessage).min(1),
  opts: ChatOptions.optional(),
});
export type ChatRequest = z.infer<typeof ChatRequest>;

export const Citation = z.object({
  doc_id: UUID,
  filename: z.string(),
  pages: z.array(z.number().int().min(1)).min(1),
  snippet: z.string().max(1000),
});
export type Citation = z.infer<typeof Citation>;

export const Usage = z.object({
  tokens_in: z.number().int().min(0),
  tokens_out: z.number().int().min(0),
});
export type Usage = z.infer<typeof Usage>;

export const ChatResponse = z.object({
  answer: z.string(),
  citations: z.array(Citation),
  usage: Usage,
  corr_id: UUID,
});
export type ChatResponse = z.infer<typeof ChatResponse>;

/** ------------------ jobs / admin ------------------ */
export const JobType = z.enum(["RAG_INDEX", "RAG_REBUILD", "RAG_RESET", "CHAT_COMPLETION"]);
export const JobState = z.enum(["queued", "running", "done", "error"]);

export const JobAccepted = z.object({
  job_id: UUID,
  corr_id: UUID,
});
export type JobAccepted = z.infer<typeof JobAccepted>;

export const JobStatusResponse = z.object({
  job_id: UUID,
  type: JobType,
  status: JobState,
  progress: z.number().min(0).max(1),
  started_at: ISODate.nullable().optional(),
  finished_at: ISODate.nullable().optional(),
  result: z.record(z.any()).nullable().optional(),
  error: z.string().nullable().optional(),
  corr_id: UUID,
});
export type JobStatusResponse = z.infer<typeof JobStatusResponse>;

export const AdminConfigResponse = z.object({
  chat_model: z.string(),
  embed_model: z.string(),
  index_dir: z.string(),
  upload_dir: z.string(),
  max_upload_mb: z.number().int().min(1),
  corr_id: UUID,
});
export type AdminConfigResponse = z.infer<typeof AdminConfigResponse>;
