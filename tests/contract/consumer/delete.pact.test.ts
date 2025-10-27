// generated-by: codex-agent 2025-02-17T13:20:00Z
import { PactV3, MatchersV3 as M } from "@pact-foundation/pact";
import fetch from "node-fetch";
import { RagDeleteRequest, RagDeleteResponse } from "../../../schemas";
import { z } from "zod";

const provider = new PactV3({
  consumer: "ChatFleet-Frontend",
  provider: "ChatFleet-API",
  logLevel: "debug",
});

describe("Pact — /admin/rag/delete", () => {
  it("deletes a RAG when confirmation matches", async () => {
    provider
      .given("RAG 'policies' exists and caller is admin")
      .uponReceiving("a delete RAG request")
      .withRequest({
        method: "POST",
        path: "/api/admin/rag/delete",
        headers: {
          Authorization: M.regex(/^Bearer .+/, "Bearer token"),
          "Content-Type": "application/json",
        },
        body: {
          rag_slug: "policies",
          confirmation: "policies",
        },
      })
      .willRespondWith({
        status: 200,
        headers: { "Content-Type": "application/json; charset=utf-8" },
        body: {
          deleted: true,
          rag_slug: "policies",
          corr_id: M.uuid(),
        },
      });

    try {
      await provider.executeTest(async (mockServer) => {
        const requestPayload: z.infer<typeof RagDeleteRequest> = {
          rag_slug: "policies",
          confirmation: "policies",
        };

        const res = await fetch(`${mockServer.url}/api/admin/rag/delete`, {
          method: "POST",
          headers: {
            Authorization: "Bearer token",
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestPayload),
        });

        const bodyText = await res.text();
        if (res.status !== 200) {
          console.error("Delete pact mock server response:", bodyText);
        }

        expect(res.status).toBe(200);
        const json = JSON.parse(bodyText);
        const parsed = RagDeleteResponse.safeParse(json);
        if (!parsed.success) {
          console.error(parsed.error.format());
        }
        expect(parsed.success).toBe(true);
      });
    } catch (error) {
      console.error("Delete pact error:", error);
      throw error;
    }
  });
});
