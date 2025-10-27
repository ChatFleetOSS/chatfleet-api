// generated-by: codex-agent 2025-02-15T00:45:00Z
import { PactV3, MatchersV3 as M } from "@pact-foundation/pact";
import fetch from "node-fetch";
import { ChatRequest, ChatResponse } from "../../../schemas";
import { z } from "zod";

const provider = new PactV3({
  consumer: "ChatFleet-Frontend",
  provider: "ChatFleet-API",
  logLevel: "debug",
});

describe("Pact — /chat", () => {
  it("returns a well-formed ChatResponse", async () => {
    provider
      .given("RAG 'policies' exists and user has access")
      .uponReceiving("a chat request")
      .withRequest({
        method: "POST",
        path: "/api/chat",
        headers: {
          Authorization: M.regex(/^Bearer .+/, "Bearer token"),
          "Content-Type": "application/json",
        },
        body: {
          rag_slug: "policies",
          messages: [{ role: "user", content: "What is parental leave?" }],
          opts: { top_k: 6, temperature: 0.2, max_tokens: 500 },
        },
      })
      .willRespondWith({
        status: 200,
        headers: { "Content-Type": "application/json; charset=utf-8" },
        body: M.like({
          answer: "Our parental leave policy...",
          citations: M.eachLike({
            doc_id: M.uuid(),
            filename: "parental_policy.pdf",
            pages: M.eachLike(4, 1),
            snippet: M.like("Eligible employees..."),
          }),
          usage: {
            tokens_in: M.integer(100),
            tokens_out: M.integer(50),
          },
          corr_id: M.uuid(),
        }),
      });

    try {
      await provider.executeTest(async (mockServer) => {
        const requestPayload: z.infer<typeof ChatRequest> = {
          rag_slug: "policies",
          messages: [
            { role: "user", content: "What is parental leave?" },
          ],
          opts: { top_k: 6, temperature: 0.2, max_tokens: 500 },
        };

        const res = await fetch(`${mockServer.url}/api/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: "Bearer token",
          },
          body: JSON.stringify(requestPayload),
        });

        const bodyText = await res.text();
        if (res.status !== 200) {
          console.error("Chat pact mock server response:", bodyText);
        }

        expect(res.status).toBe(200);
        const json = JSON.parse(bodyText);
        const parsed = ChatResponse.safeParse(json);
        if (!parsed.success) {
          console.error(parsed.error.format());
        }
        expect(parsed.success).toBe(true);
      });
    } catch (error) {
      console.error("Chat pact error:", error);
      throw error;
    }
  });
});
