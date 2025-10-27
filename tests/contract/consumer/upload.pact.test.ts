// generated-by: codex-agent 2025-02-15T00:45:00Z
import { PactV3, MatchersV3 as M } from "@pact-foundation/pact";
import fetch from "node-fetch";
import { RagUploadAccepted } from "../../../schemas";

const UPLOAD_BOUNDARY = "--------------------------PactBoundary";
const MULTIPART_BODY =
  `--${UPLOAD_BOUNDARY}\r\n` +
  `Content-Disposition: form-data; name="rag_slug"\r\n\r\n` +
  `policies\r\n` +
  `--${UPLOAD_BOUNDARY}\r\n` +
  `Content-Disposition: form-data; name="files"; filename="handbook.pdf"\r\n` +
  `Content-Type: application/pdf\r\n\r\n` +
  `%PDF-1.7 placeholder%\r\n` +
  `--${UPLOAD_BOUNDARY}--\r\n`;

const provider = new PactV3({
  consumer: "ChatFleet-Frontend",
  provider: "ChatFleet-API",
  logLevel: "debug",
});

describe("Pact — /rag/upload", () => {
  it("accepts a multipart PDF upload and returns a job_id", async () => {
    provider
      .given("RAG 'policies' exists and caller is admin")
      .uponReceiving("a file upload")
      .withRequest({
        method: "POST",
        path: "/api/rag/upload",
        headers: {
          Authorization: M.regex(/^Bearer .+/, "Bearer token"),
          "Content-Type": `multipart/form-data; boundary=${UPLOAD_BOUNDARY}`,
        },
        body: MULTIPART_BODY,
      })
      .willRespondWith({
        status: 202,
        headers: { "Content-Type": "application/json; charset=utf-8" },
        body: {
          job_id: M.uuid(),
          accepted: M.eachLike("handbook.pdf"),
          skipped: M.eachLike("duplicate.pdf"),
          rag_slug: "policies",
          corr_id: M.uuid(),
        },
      });

    try {
      await provider.executeTest(async (mockServer) => {
        const res = await fetch(`${mockServer.url}/api/rag/upload`, {
          method: "POST",
          headers: {
            Authorization: "Bearer token",
            "Content-Type": `multipart/form-data; boundary=${UPLOAD_BOUNDARY}`,
            "Content-Length": Buffer.byteLength(MULTIPART_BODY).toString(),
          },
          body: MULTIPART_BODY,
        });

        const bodyText = await res.text();
        if (res.status !== 202) {
          console.error("Upload pact mock server response:", bodyText);
        }

        expect(res.status).toBe(202);
        const json = JSON.parse(bodyText);
        const parsed = RagUploadAccepted.safeParse(json);
        expect(parsed.success).toBe(true);
      });
    } catch (error) {
      console.error("Upload pact error:", error);
      throw error;
    }
  });
});
