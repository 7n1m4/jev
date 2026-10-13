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
		// The local service is free and needs no key, but the extension
		// provider-registration contract REQUIRES one: ModelRegistry
		// .registerProvider() validates through validateProviderConfiguration()
		// in "runtime-register" mode (packages/coding-agent/src/config/
		// models-config.ts:77-88), where a provider that declares custom models
		// must supply `apiKey` or `oauth` or registration throws
		//   Provider typesafe: "apiKey" or "oauth" is required when defining models.
		// which is fatal -- the whole extension fails to load and no provider is
		// registered at all.
		//
		// The literal is therefore a placeholder credential, NOT a secret. Two
		// things keep it out of the request:
		//   * `authHeader: false` below, so the config header resolver never
		//     synthesizes an Authorization header (createConfigHeaderResolver
		//     only emits one when authHeader === true AND apiKeyConfig is set
		//     -- resolve-config-value.ts:193);
		//   * the native judgment wire sends its OWN bearer, and ChainJudge
		//     resolves that credential from the auth store, which this literal
		//     populates so the judge chain's "no API key" branch is not taken
		//     (packages/coding-agent/src/judgment/index.ts:338).
		// The service ignores the Authorization header entirely, so a real
		// value here would neither help nor hurt -- it exists purely to satisfy
		// the registration contract.
		apiKey: "local",
		// Explicit, and redundant with the apiKey above only for a reader: no
		// Authorization header is attached to any request.
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
