const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function parseResponse(response) {
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    const message = data?.detail || data?.message || `Request failed with status ${response.status}`
    throw new Error(message)
  }
  return data
}

export async function checkHealth() {
  return parseResponse(await fetch(`${API_BASE_URL}/api/health`))
}

export async function listWorkspaces() {
  return parseResponse(await fetch(`${API_BASE_URL}/api/workspaces`))
}

export async function uploadWorkspace(file, title) {
  const form = new FormData()
  form.append('file', file)
  if (title.trim()) form.append('title', title.trim())
  return parseResponse(await fetch(`${API_BASE_URL}/api/workspaces`, {
    method: 'POST',
    body: form,
  }))
}

export async function askWorkspace(workspaceId, payload) {
  return parseResponse(await fetch(`${API_BASE_URL}/api/workspaces/${workspaceId}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }))
}

export async function deleteWorkspace(workspaceId) {
  return parseResponse(await fetch(`${API_BASE_URL}/api/workspaces/${workspaceId}`, {
    method: 'DELETE',
  }))
}
