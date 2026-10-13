/**
 * Offline load check for the jev-local extension.
 *
 * Executes the REAL extension source with a stubbed ExtensionAPI and verifies it
 * registers the local provider correctly, without starting an omp session. This
 * validates the export shape, the registerProvider payload, and the failure-free
 * path when the service is NOT running.
 */
import { pathToFileURL } from "node:url";
import { pathToFileURL as _p } from "node:url";

const EXT = process.argv[2] ?? "/home/vi/work/jev-local/extension/index.ts";
const BASE = process.env.JEV_BASE_URL ?? "";

const calls = [];
const logs = [];

const pi = {
	zod: {},
	registerProvider(name, config) {
		calls.push({ name, config });
	},
	unregisterProvider(name) {
		calls.push({ unregister: name });
	},
	registerTool() {},
	registerCommand() {},
	registerHook() {},
	registerSkill() {},
	addAutocompleteProvider() {},
	logger: {
		info: (...a) => logs.push(a),
		debug: () => {},
		warn: (...a) => logs.push(a),
		error: (...a) => logs.push(a),
	},
	events: { on() {}, emit() {} },
};

const mod = await import(pathToFileURL(EXT).href);
const fn = mod.default ?? mod.register ?? (typeof mod === "function" ? mod : null);

let failures = [];
function check(label, cond, detail) {
	const ok = !!cond;
	console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` -- ${detail}` : ""}`);
	if (!ok) failures.push(label);
}

if (typeof fn !== "function") {
	console.log("FAIL  extension exports a register function");
	console.log("     exports:", Object.keys(mod));
	process.exit(1);
}
check("exports default register function (pi: ExtensionAPI)", true, fn.name || "anonymous");

let threw = null;
try {
	fn(pi);
} catch (e) {
	threw = e;
}
check("loads without throwing when the service is NOT running", threw === null,
	threw ? String(threw?.message ?? threw) : "");

check("registerProvider called exactly once", calls.length === 1,
	calls.length === 1 ? "1 call" : `${calls.length} calls`);

if (calls.length === 1) {
	const { name, config } = calls[0];
	check("provider name is 'typesafe'", name === "typesafe", name);
	check("api is 'typesafe' (routes NativeJudge -> POST {baseUrl}/v1/systemone)",
		config.api === "typesafe", String(config.api));
	check("explicit baseUrl set (beats TYPESAFE_BASE_URL env)",
		typeof config.baseUrl === "string" && config.baseUrl.length > 0, config.baseUrl);
	check("baseUrl is loopback", /^https?:\/\/127\.0\.0\.1:\d+$/.test(config.baseUrl ?? ""),
		config.baseUrl);
	check("authHeader disabled", config.authHeader === false, String(config.authHeader));
	check("exactly one model registered", Array.isArray(config.models) && config.models.length === 1,
		config.models?.length);
	const m = config.models?.[0];
	check("model id matches the service's /v1/models name",
		m?.id === "jev-style-0.8b-decision-v3", m?.id);
	check("model kind is 'judge' (judge-role eligible)", m?.kind === "judge", m?.kind);
	check("model api inherits 'typesafe'", m?.api === "typesafe" || config.api === "typesafe",
		String(m?.api ?? config.api));
	check("model baseUrl matches the provider baseUrl", m?.baseUrl === config.baseUrl,
		m?.baseUrl);
	check("reasoning false", m?.reasoning === false, String(m?.reasoning));
	check("input is ['text']", Array.isArray(m?.input) && m?.input.length === 1 && m?.input[0] === "text",
		JSON.stringify(m?.input));
	check("cost is all zero (local, free)",
		m?.cost && ["input", "output", "cacheRead", "cacheWrite"].every(k => m.cost[k] === 0),
		JSON.stringify(m?.cost));
	check("maxTokens is a positive number (ProviderConfigInput requires number)",
		typeof m?.maxTokens === "number" && m.maxTokens > 0, String(m?.maxTokens));
	check("contextWindow positive", typeof m?.contextWindow === "number" && m.contextWindow > 0,
		String(m?.contextWindow));
	check("no streamSimple override (uses omp's built-in typesafe transport)",
		typeof config.streamSimple !== "function");
}

check("logged an info line about the judge role",
	logs.length > 0, logs.length ? String(logs[0][0]).slice(0, 80) : "");

console.log("");
if (failures.length) {
	console.log(`RESULT: ${failures.length} CHECK(S) FAILED: ${failures.join(", ")}`);
	process.exit(1);
}
console.log("RESULT: extension loads and registers correctly (offline, no omp session)");
