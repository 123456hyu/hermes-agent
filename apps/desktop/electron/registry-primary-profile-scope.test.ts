import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

import { pathForRegistryBackendRequest, resolveRegistryRequestPath } from './connection-config'

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

describe('registry local delegate keeps per-request profile scope (#119411)', () => {
  it('scopes a delegated non-primary model write with ?profile=<profile>', () => {
    // Settings → Models picker on a non-primary profile: the delegate branch
    // resolves the backend WITHOUT the request, so the descriptor alone
    // cannot vouch for scope. Dispatch must scope via the v1 table instead
    // of leaving a bare path the server resolves to its launch home.
    expect(
      resolveRegistryRequestPath('/api/model/set', 'ro', { registryLocalDelegate: true }, {
        requestMethod: 'POST',
        requestPath: '/api/model/set'
      })
    ).toBe('/api/model/set?profile=ro')
  })

  it('scopes a delegated non-primary model read with ?profile=<profile>', () => {
    expect(
      resolveRegistryRequestPath('/api/model/info', 'atenea', { registryLocalDelegate: true }, {
        requestMethod: 'GET',
        requestPath: '/api/model/info'
      })
    ).toBe('/api/model/info?profile=atenea')
  })

  it('scopes the delegated primary on a scopable route (launch home may differ)', () => {
    // resolveProfileBackendRoute drops the descriptor tag for the primary
    // when resolved without a request, so a sharedPrimary-flag check alone
    // leaves the primary bare — the #118432 case through the delegate.
    expect(
      resolveRegistryRequestPath('/api/config', 'default', { registryLocalDelegate: true }, {
        primaryProfile: 'default',
        requestMethod: 'GET',
        requestPath: '/api/config'
      })
    ).toBe('/api/config?profile=default')
  })

  it('leaves a delegated unscopable mutating request bare', () => {
    // No ?profile= the handler reads: inventing one advertises a scope that
    // is not doing the work.
    expect(
      resolveRegistryRequestPath('/api/custom-thing', 'ro', { registryLocalDelegate: true }, {
        requestMethod: 'POST',
        requestPath: '/api/custom-thing'
      })
    ).toBe('/api/custom-thing')
  })

  it('keeps the non-delegate registry route byte-identical', () => {
    expect(resolveRegistryRequestPath('/api/skills', 'acme', { sharedRemote: true }, {})).toBe(
      '/api/skills?profile=acme'
    )
    expect(
      resolveRegistryRequestPath('/api/skills', 'acme', { sharedRemote: false, remoteProfile: null }, {})
    ).toBe('/api/skills')
  })

  it('wires the delegate flag from ensureRegistryBackend into dispatch', () => {
    const delegateStart = mainSource.indexOf('if (localRoute.delegate)')
    expect(delegateStart).toBeGreaterThan(-1)
    const delegateBranch = mainSource.slice(delegateStart, delegateStart + 400)
    expect(delegateBranch).toContain('registryLocalDelegate: true')

    const dispatchStart = mainSource.indexOf('async function dispatchRegistryApiRequest(')
    expect(dispatchStart).toBeGreaterThan(-1)
    const dispatchBody = mainSource.slice(dispatchStart, dispatchStart + 2500)
    expect(dispatchBody).toContain('resolveRegistryRequestPath(')
    expect(dispatchBody).toContain('profileRouteOptions(requestProfile, request)')
  })
})
