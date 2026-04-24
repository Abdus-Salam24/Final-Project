// API endpoint configuration
const rawApiBase = import.meta.env.VITE_API_URL;

export const API = rawApiBase.replace(/\/+$/, '');

interface ApiRequestOptions {
	fallbackPaths?: string[];
}

export function buildApiUrl(path: string, base: string = API) {
	const normalizedBase = base.replace(/\/+$/, '');
	const normalizedPath = path.startsWith('/') ? path : `/${path}`;

	if (!normalizedBase) {
		return normalizedPath;
	}

	if (normalizedBase === '/api' && normalizedPath.startsWith('/api/')) {
		return normalizedPath;
	}

	if (normalizedBase.endsWith('/api') && normalizedPath.startsWith('/api/')) {
		return `${normalizedBase}${normalizedPath.slice(4)}`;
	}

	return `${normalizedBase}${normalizedPath}`;
}

function buildCandidatePaths(path: string, fallbackPaths: string[] = []) {
	const normalizedPath = path.startsWith('/') ? path : `/${path}`;
	const candidates = [normalizedPath];

	if (normalizedPath.startsWith('/api/')) {
		candidates.push(normalizedPath.slice(4));
	} else {
		candidates.push(`/api${normalizedPath}`);
	}

	for (const fallbackPath of fallbackPaths) {
		candidates.push(fallbackPath.startsWith('/') ? fallbackPath : `/${fallbackPath}`);
	}

	return Array.from(new Set(candidates));
}

export async function apiFetch(path: string, init?: RequestInit, options?: ApiRequestOptions) {
	const errors: string[] = [];
	const candidates = buildCandidatePaths(path, options?.fallbackPaths);

	for (const candidatePath of candidates) {
		const url = buildApiUrl(candidatePath);
		try {
			const response = await fetch(url, init);
			if (response.ok) {
				return response;
			}

			const detail = await response.text().catch(() => '');
			errors.push(`${candidatePath} -> HTTP ${response.status}${detail ? `: ${detail}` : ''}`);
		} catch (error: unknown) {
			errors.push(`${candidatePath} -> ${error instanceof Error ? error.message : String(error)}`);
		}
	}

	throw new Error(`All API endpoints failed for ${path}. ${errors.join(' | ')}`);
}

export async function apiGetJson<T>(path: string, options?: ApiRequestOptions): Promise<T> {
	const response = await apiFetch(path, { method: 'GET' }, options);
	return response.json() as Promise<T>;
}

export async function apiPostJson<T>(path: string, body: unknown, options?: ApiRequestOptions): Promise<T> {
	const response = await apiFetch(path, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	}, options);
	return response.json() as Promise<T>;
}

export async function apiPostForm<T>(path: string, form: FormData, options?: ApiRequestOptions): Promise<T> {
	const response = await apiFetch(path, {
		method: 'POST',
		body: form,
	}, options);
	return response.json() as Promise<T>;
}
