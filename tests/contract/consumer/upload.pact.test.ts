// generated-by: codex-agent 2025-02-15T00:45:00Z
import { PactV3, MatchersV3 as M } from "@pact-foundation/pact";
import fetch from "node-fetch";
import { RagUploadAccepted } from "../../../schemas";

const UPLOAD_BOUNDARY = "--------------------------PactBoundary";
const RAG_SLUG = "policies";

type UploadCase = {
  label: string;
  filename: string;
  mime: string;
  contents: string;
};

const uploadCases: UploadCase[] = [
  {
    label: "PDF",
    filename: "handbook.pdf",
    mime: "application/pdf",
    contents: "%PDF-1.7 pact fixture",
  },
  {
    label: "TXT",
    filename: "notes.txt",
    mime: "text/plain",
    contents: "Plain text pact fixture",
  },
];

const provider = new PactV3({
  consumer: "ChatFleet-Frontend",
  provider: "ChatFleet-API",
  logLevel: "debug",
});

function multipartBody({ filename, mime, contents }: UploadCase) {
  return (
    `--${UPLOAD_BOUNDARY}\r\n` +
    `Content-Disposition: form-data; name="rag_slug"\r\n\r\n` +
    `${RAG_SLUG}\r\n` +
    `--${UPLOAD_BOUNDARY}\r\n` +
    `Content-Disposition: form-data; name="files"; filename="${filename}"\r\n` +
    `Content-Type: ${mime}\r\n\r\n` +
    `${contents}\r\n` +
    `--${UPLOAD_BOUNDARY}--\r\n`
  );
}

describe("Pact — /rag/upload", () => {
  it.each(uploadCases)(
    "accepts a multipart $label upload and returns a job_id",
    async (uploadCase) => {
      const body = multipartBody(uploadCase);

      provider
        .given("RAG 'policies' exists and caller is admin")
        .uponReceiving(`a ${uploadCase.label} file upload`)
        .withRequest({
          method: "POST",
          path: "/api/rag/upload",
          headers: {
            Authorization: M.regex(/^Bearer .+/, "Bearer token"),
            "Content-Type": `multipart/form-data; boundary=${UPLOAD_BOUNDARY}`,
          },
          body,
        })
        .willRespondWith({
          status: 202,
          headers: { "Content-Type": "application/json; charset=utf-8" },
          body: {
            job_id: M.uuid(),
            accepted: M.eachLike(uploadCase.filename),
            skipped: M.constrainedArrayLike("duplicate.pdf", 0, 20, 1),
            rag_slug: RAG_SLUG,
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
              "Content-Length": body.length.toString(),
            },
            body,
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
    }
  );
});
