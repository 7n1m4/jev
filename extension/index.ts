/**
 * jev-local: point oh-my-pi's `typesafe` judge provider at a LOCAL Jev-Style
 * 0.8B decision service (CPU-only llama.cpp), running on this machine.
 *
 * Why an extension and not the env var alone: a catalog row's explicit baseUrl
 * beats the TYPESAFE_BASE_URL environment variable, so setting TYPESAFE_BASE_URL
 * is NOT sufficient to retarget the judge role. Registering the provider here
 * writes the explicit baseUrl that wins the precedence order.
 *
 * This extension is safe when the service is not running:
 *   - registration is pure in-memory metadata; nothing is dialled at load time;
 *   - the `judge` role is NOT reassigned (see "Judge role" below) -- the extension
 *     only supplies the local provider, so an omp session keeps working unchanged
 *     until the operator opts in by editing `modelRoles.judge`.
 */
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";

/** Local service contract: loopback only (see service/README.md for the port rationale). */
const DEFAULT_BASE_URL = "http://127.0.0.1:61021";

/** The model name the local service reports from GET /v1/models. */
const MODEL_ID = "jev-style-0.8b-decision-v3";

/** Overrides, so the extension can be pointed elsewhere without editing source. */
const BASE_URL = (
	process.env.JEV_BASE_URL
	?? process.env.JEV_LOCAL_BASE_URL
	?? DEFAULT_BASE_URL
).replace(/\/+$/, "");

export default function (pi: ExtensionAPI) {
	pi.registerProvider("typesafe", {
		baseUrl: BASE_URL,
		api: "typesafe",
		// The local service is free and needs no key. `authHeader: false` keeps any
		// Authorization header omp would otherwise add from a stale key out of the
		// request; the service accepts any bearer anyway.
		authHeader: false,
		models: [
			{
				id: MODEL_ID,
				name: "Jev-Style 0.8B Decision v3 (local, CPU)",
				api: "typesafe",
				kind: "judge",
				reasoning: false,
				input: ["text"],
				supportsTools: false,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 262144,
				// Judge decisions return a small typed readout, not text; a positive
				// non-zero cap is required by ProviderConfigInput (maxTokens: number).
				maxTokens: 1,
				baseUrl: BASE_URL,
			},
		],
	});

	pi.logger.info(
		`jev-local: registered local typesafe provider at ${BASE_URL} (${MODEL_ID}). `
		+ `Set modelRoles.judge to "typesafe/${MODEL_ID}" in config.yml to route the judge role here.`,
	);
}
