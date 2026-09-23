import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { pathForRegistryBackendRequest } from './connection-config'

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

describe('primary-remote descriptor reuse keeps profile scope', () => {
  it('scopes a shared-remote request with ?profile=<profile>', () => {
    expect(pathForRegistryBackendRequest('/api/skills', 'acme', { sharedRemote: true })).toBe(
      '/api/skills?profile=acme'
    )
  })

  it('does not add a profile query when the backend is not shared-remote', () => {
    // An isolated backend owns one profile; the router must not invent a scope.
    expect(pathForRegistryBackendRequest('/api/skills', 'acme', { sharedRemote: false, remoteProfile: null })).toBe(
      '/api/skills'
    )
  })

  it('marks the reused primary-remote descriptor sharedRemote so the router scopes it', () => {
    const branchStart = mainSource.indexOf("if (id === registry.primary && source.kind !== 'local'")
    expect(branchStart).toBeGreaterThan(-1)
    const branch = mainSource.slice(branchStart, branchStart + 800)

    // The reuse branch must decorate the ambient primary descriptor with
    // sharedRemote: true, matching the explicit shared-remote connection path.
    expect(branch).toContain('const primaryDescriptor = await ensureBackend(profile, { passive })')
    expect(branch).toContain('registrySourceOwnsPrimaryBackend(registry, id, primaryDescriptor)')
    expect(branch).toContain('sharedRemote: true')
  })
})

describe('registry local delegate keeps per-request profile scope (#119411, #119415, #119894)', () => {
  const delegate = { registryLocalDelegate: true }

  it('scopes delegated reads and writes for another profile through the v1 route table', () => {
    // Settings → Models / Capabilities on a non-primary profile through the
    // registry 'local' pin: the shared host backend resolves a bare path to its
    // LAUNCH home, so the selected profile must ride the wire.
    expect(
      pathForRegistryBackendRequest('/api/model/set', 'ro', delegate, { requestMethod: 'POST' })
    ).toBe('/api/model/set?profile=ro')
    expect(
      pathForRegistryBackendRequest('/api/skills/toggle', 'orchestrator', delegate, { requestMethod: 'PUT' })
    ).toBe('/api/skills/toggle?profile=orchestrator')
    // The primary too: the attached host backend may have booted under another
    // profile's home (#118432 through the delegate).
    expect(
      pathForRegistryBackendRequest('/api/config', 'default', delegate, { primaryProfile: 'default' })
    ).toBe('/api/config?profile=default')
    // An explicit cross-profile selector already on the path wins.
    expect(pathForRegistryBackendRequest('/api/skills?profile=all', 'ro', delegate, {})).toBe(
      '/api/skills?profile=all'
    )
  })

  it('never invents a scope the handler does not read, and leaves non-delegates unchanged', () => {
    // A mutation the server cannot scope keeps its pooled backend (case 6):
    // a `?profile=` here would advertise a scope that is not doing the work.
    expect(
      pathForRegistryBackendRequest('/api/custom-thing', 'ro', delegate, { requestMethod: 'POST' })
    ).toBe('/api/custom-thing')
    expect(pathForRegistryBackendRequest('/api/skills', 'acme', { sharedRemote: true }, {})).toBe(
      '/api/skills?profile=acme'
    )
    expect(pathForRegistryBackendRequest('/api/skills', 'acme', { sharedRemote: false, remoteProfile: null })).toBe(
      '/api/skills'
    )
  })
})
