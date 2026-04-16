// generated-by: codex-agent 2025-02-17T13:30:00Z
import { Verifier } from "@pact-foundation/pact";
import fs from "fs";
import path from "path";
import { execFile } from "child_process";
import { promisify } from "util";

const execFileAsync = promisify(execFile);
const repoRoot = path.resolve(process.cwd());

const ADMIN_EMAIL = "admin@chatfleet.local";
const ADMIN_PASSWORD = "adminpass";
const USER_EMAIL = "pact-user@chatfleet.local";
const USER_PASSWORD = "userpass123";
const RAG_SLUG = "policies";

const tokens = {
  admin: null,
  user: null,
};

function applyBearer(req, token) {
  if (!token) {
    return;
  }
  const bearer = `Bearer ${token}`;
  req.headers = req.headers ?? {};
  req.setHeader?.("Authorization", bearer);
  req.setHeader?.("authorization", bearer);
  req.addHeader?.("Authorization", bearer);
  req.addHeader?.("authorization", bearer);
  req.headers["Authorization"] = bearer;
  req.headers["authorization"] = bearer;
}

async function httpFetch(input, init) {
  if (typeof fetch === "function") {
    return fetch(input, init);
  }
  const mod = await import("node-fetch");
  return mod.default(input, init);
}

async function seedAdminUser() {
  const setupPython = process.env.pythonLocation
    ? path.join(
        process.env.pythonLocation,
        process.platform === "win32" ? "python.exe" : "bin/python"
      )
    : null;
  const venvPython = path.join(repoRoot, ".venv", "bin", "python");
  const pythonBin =
    process.env.PYTHON ??
    (setupPython && fs.existsSync(setupPython)
      ? setupPython
      : fs.existsSync(venvPython)
        ? venvPython
        : "python3");
  try {
    await execFileAsync(pythonBin, ["scripts/seed_admin.py"], {
      env: {
        ...process.env,
        PYTHONPATH: process.env.PYTHONPATH
          ? `${process.env.PYTHONPATH}:${repoRoot}`
          : repoRoot,
      },
    });
  } catch (error) {
    if (error.code === "ENOENT" && pythonBin !== "python") {
      await execFileAsync("python", ["scripts/seed_admin.py"], {
        env: {
          ...process.env,
          PYTHONPATH: process.env.PYTHONPATH
            ? `${process.env.PYTHONPATH}:${repoRoot}`
            : repoRoot,
        },
      });
      return;
    }
    console.error(`✖ seed_admin.py failed via ${pythonBin}`);
    if (error.stdout) {
      console.error(String(error.stdout).trim());
    }
    if (error.stderr) {
      console.error(String(error.stderr).trim());
    }
    throw error;
  }
}

async function login(baseUrl, email, password) {
  const res = await httpFetch(`${baseUrl}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Login failed for ${email}: ${res.status} ${body}`);
  }
  const json = await res.json();
  return json.token;
}

async function ensureRag(baseUrl, adminToken) {
  const res = await httpFetch(`${baseUrl}/api/rag`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${adminToken}`,
    },
    body: JSON.stringify({
      slug: RAG_SLUG,
      name: "Policies",
      description: "Pact verification knowledge base",
    }),
  });
  if (res.ok || res.status === 409) {
    return;
  }
  const body = await res.text();
  throw new Error(`Failed to ensure RAG: ${res.status} ${body}`);
}

async function registerUser(baseUrl, email, password) {
  const res = await httpFetch(`${baseUrl}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      email,
      name: "Pact User",
      password,
    }),
  });
  if (res.status === 201 || res.status === 400) {
    return;
  }
  const body = await res.text();
  throw new Error(`Failed to register user: ${res.status} ${body}`);
}

async function grantRagAccess(baseUrl, adminToken, email) {
  const res = await httpFetch(`${baseUrl}/api/rag/users/add`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${adminToken}`,
    },
    body: JSON.stringify({
      rag_slug: RAG_SLUG,
      email,
    }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Failed to grant access: ${res.status} ${body}`);
  }
}

async function ensureAdminState(baseUrl) {
  await seedAdminUser();
  const adminToken = await login(baseUrl, ADMIN_EMAIL, ADMIN_PASSWORD);
  tokens.admin = adminToken;
  await ensureRag(baseUrl, adminToken);
}

async function ensureUserState(baseUrl) {
  await ensureAdminState(baseUrl);
  await registerUser(baseUrl, USER_EMAIL, USER_PASSWORD);
  await grantRagAccess(baseUrl, tokens.admin, USER_EMAIL);
  tokens.user = await login(baseUrl, USER_EMAIL, USER_PASSWORD);
}

async function main() {
  const baseUrl = process.env.PROVIDER_BASE_URL ?? "http://127.0.0.1:8000";
  const pactsDir = path.resolve(process.cwd(), "pacts");

  console.log(`Using provider base URL: ${baseUrl}`);

  const verifier = new Verifier({
    providerBaseUrl: baseUrl,
    pactUrls: [path.join(pactsDir, "ChatFleet-Frontend-ChatFleet-API.json")],
    publishVerificationResult: false,
    providerVersion: process.env.PROVIDER_VERSION ?? "local-dev",
    stateHandlers: {
      "RAG 'policies' exists and caller is admin": async () => {
        await ensureAdminState(baseUrl);
      },
      "RAG 'policies' exists and user has access": async () => {
        await ensureUserState(baseUrl);
      },
    },
    requestFilter: (req, _res, next) => {
      try {
        const pathLower = String(req.path || req.url || "").toLowerCase();
        if (pathLower.includes("/api/rag/upload") && tokens.admin) {
          applyBearer(req, tokens.admin);
        }
        if (pathLower.includes("/api/rag/delete") && tokens.admin) {
          applyBearer(req, tokens.admin);
        }
        if (pathLower.includes("/api/chat") && tokens.user) {
          applyBearer(req, tokens.user);
        }
        next();
      } catch (error) {
        console.error("✖ Request filter failed", error);
        next(error);
      }
    },
  });

  try {
    const output = await verifier.verifyProvider();
    console.log("✔ Pact verification complete");
    console.log(output);
  } catch (error) {
    console.error("✖ Pact verification failed", error);
    process.exit(1);
  }
}

main().catch((error) => {
  console.error("✖ Pact verifier crashed", error);
  process.exit(1);
});
