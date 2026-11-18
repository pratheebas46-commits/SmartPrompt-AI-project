export type PublicRoute = 'home' | 'login' | 'register' | 'app';

export interface AuthUser {
  userId: string;
  fullName: string;
  username: string;
  email: string;
  createdAt: string;
}

export interface AuthSession {
  token: string;
  user: AuthUser;
  authSource: 'local' | 'api';
  expiresAt: string;
}

interface StoredUserRecord {
  userId: string;
  fullName: string;
  username: string;
  email: string;
  passwordHash: string;
  createdAt: string;
}

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface RegisterInput {
  fullName: string;
  username: string;
  email: string;
  password: string;
  confirmPassword: string;
}

export interface AuthActionResult {
  ok: boolean;
  message: string;
  user?: AuthUser;
  session?: AuthSession;
  fieldErrors?: Partial<Record<keyof RegisterInput | 'identifier', string>>;
}

export const AUTH_USERS_STORAGE_KEY = 'smartprompt_ai_auth_users_v1';
export const AUTH_SESSION_STORAGE_KEY = 'smartprompt_ai_auth_session_v1';

const USERNAME_REGEX = /^[a-zA-Z0-9_.-]{3,32}$/;
const EMAIL_REGEX = /^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/;
const PBKDF2_ITERATIONS = 100_000;
const SESSION_TTL_MS = 72 * 60 * 60 * 1000;

function getDefaultStorage(): StorageLike | null {
  try {
    if (typeof globalThis !== 'undefined' && 'localStorage' in globalThis && globalThis.localStorage) {
      return globalThis.localStorage;
    }
  } catch {
    return null;
  }
  return null;
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

function hexToBytes(hex: string): Uint8Array {
  const clean = hex.trim();
  const out = new Uint8Array(clean.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(clean.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

function randomHex(byteCount: number): string {
  const bytes = new Uint8Array(byteCount);
  if (typeof globalThis !== 'undefined' && globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < byteCount; i++) {
      bytes[i] = Math.floor(Math.random() * 256);
    }
  }
  return bytesToHex(bytes);
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}

// Pure-TS SHA-256 fallback used only when Web Crypto SubtleCrypto is unavailable (e.g. non-secure context)
function sha256HexFallback(input: string): string {
  const utf8 = new TextEncoder().encode(input);
  const K = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ]);
  const H = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ]);
  const bitLen = utf8.length * 8;
  const paddedLen = Math.ceil((utf8.length + 9) / 64) * 64;
  const padded = new Uint8Array(paddedLen);
  padded.set(utf8);
  padded[utf8.length] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(paddedLen - 4, bitLen >>> 0, false);
  view.setUint32(paddedLen - 8, Math.floor(bitLen / 0x100000000), false);

  const W = new Uint32Array(64);
  const rotr = (x: number, n: number) => (x >>> n) | (x << (32 - n));

  for (let offset = 0; offset < paddedLen; offset += 64) {
    for (let i = 0; i < 16; i++) W[i] = view.getUint32(offset + i * 4, false);
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(W[i - 15], 7) ^ rotr(W[i - 15], 18) ^ (W[i - 15] >>> 3);
      const s1 = rotr(W[i - 2], 17) ^ rotr(W[i - 2], 19) ^ (W[i - 2] >>> 10);
      W[i] = (W[i - 16] + s0 + W[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = H;
    for (let i = 0; i < 64; i++) {
      const S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + K[i] + W[i]) >>> 0;
      const S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;
      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }
    H[0] = (H[0] + a) >>> 0;
    H[1] = (H[1] + b) >>> 0;
    H[2] = (H[2] + c) >>> 0;
    H[3] = (H[3] + d) >>> 0;
    H[4] = (H[4] + e) >>> 0;
    H[5] = (H[5] + f) >>> 0;
    H[6] = (H[6] + g) >>> 0;
    H[7] = (H[7] + h) >>> 0;
  }

  return Array.from(H)
    .map((w) => w.toString(16).padStart(8, '0'))
    .join('');
}

export async function hashPassword(password: string, existingSaltHex?: string): Promise<string> {
  const saltHex = existingSaltHex ?? randomHex(16);
  const subtle = typeof globalThis !== 'undefined' ? globalThis.crypto?.subtle : undefined;
  if (subtle) {
    const keyMaterial = await subtle.importKey(
      'raw',
      new TextEncoder().encode(password),
      'PBKDF2',
      false,
      ['deriveBits']
    );
    const derivedBits = await subtle.deriveBits(
      {
        name: 'PBKDF2',
        salt: hexToBytes(saltHex) as unknown as BufferSource,
        iterations: PBKDF2_ITERATIONS,
        hash: 'SHA-256',
      },
      keyMaterial,
      256
    );
    return `pbkdf2_sha256$${PBKDF2_ITERATIONS}$${saltHex}$${bytesToHex(new Uint8Array(derivedBits))}`;
  }

  let digest = sha256HexFallback(`${saltHex}:${password}`);
  for (let i = 0; i < 200; i++) {
    digest = sha256HexFallback(`${digest}:${saltHex}:${password}`);
  }
  return `iter_sha256$200$${saltHex}$${digest}`;
}

export async function verifyPassword(password: string, storedHash: string): Promise<boolean> {
  const parts = storedHash.split('$');
  if (parts.length !== 4) return false;
  const [scheme, iterStr, saltHex, expectedHex] = parts;
  const iterations = Number(iterStr);
  if (!Number.isFinite(iterations) || !saltHex || !expectedHex) return false;

  if (scheme === 'pbkdf2_sha256') {
    const subtle = typeof globalThis !== 'undefined' ? globalThis.crypto?.subtle : undefined;
    if (!subtle) return false;
    const keyMaterial = await subtle.importKey(
      'raw',
      new TextEncoder().encode(password),
      'PBKDF2',
      false,
      ['deriveBits']
    );
    const derivedBits = await subtle.deriveBits(
      {
        name: 'PBKDF2',
        salt: hexToBytes(saltHex) as unknown as BufferSource,
        iterations,
        hash: 'SHA-256',
      },
      keyMaterial,
      256
    );
    return constantTimeEqual(bytesToHex(new Uint8Array(derivedBits)), expectedHex);
  }

  if (scheme === 'iter_sha256') {
    let digest = sha256HexFallback(`${saltHex}:${password}`);
    for (let i = 0; i < iterations; i++) {
      digest = sha256HexFallback(`${digest}:${saltHex}:${password}`);
    }
    return constantTimeEqual(digest, expectedHex);
  }

  return false;
}

export function validateRegistrationInput(input: RegisterInput): {
  valid: boolean;
  fieldErrors: Partial<Record<keyof RegisterInput, string>>;
  normalized: { fullName: string; username: string; email: string };
} {
  const fullName = input.fullName.trim();
  const username = input.username.trim().toLowerCase();
  const email = input.email.trim().toLowerCase();
  const password = input.password;
  const confirmPassword = input.confirmPassword;

  const fieldErrors: Partial<Record<keyof RegisterInput, string>> = {};

  if (!fullName) {
    fieldErrors.fullName = 'Full name is required.';
  } else if (fullName.length < 2) {
    fieldErrors.fullName = 'Full name must be at least 2 characters.';
  }

  if (!username) {
    fieldErrors.username = 'Username is required.';
  } else if (!USERNAME_REGEX.test(username)) {
    fieldErrors.username = 'Username must be 3–32 characters (letters, numbers, ".", "-", "_").';
  }

  if (!email) {
    fieldErrors.email = 'Email address is required.';
  } else if (!EMAIL_REGEX.test(email)) {
    fieldErrors.email = 'Please enter a valid email address.';
  }

  if (!password) {
    fieldErrors.password = 'Password is required.';
  } else if (password.length < 8) {
    fieldErrors.password = 'Password must be at least 8 characters long.';
  } else if (!/[A-Za-z]/.test(password) || !/\d/.test(password)) {
    fieldErrors.password = 'Password must include at least one letter and one number.';
  }

  if (!confirmPassword) {
    fieldErrors.confirmPassword = 'Please confirm your password.';
  } else if (password !== confirmPassword) {
    fieldErrors.confirmPassword = 'Passwords do not match.';
  }

  return {
    valid: Object.keys(fieldErrors).length === 0,
    fieldErrors,
    normalized: { fullName, username, email },
  };
}

function readStoredUsers(storage: StorageLike | null): StoredUserRecord[] {
  if (!storage) return [];
  try {
    const raw = storage.getItem(AUTH_USERS_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as StoredUserRecord[]) : [];
  } catch {
    return [];
  }
}

function writeStoredUsers(storage: StorageLike | null, users: StoredUserRecord[]): void {
  if (!storage) return;
  try {
    storage.setItem(AUTH_USERS_STORAGE_KEY, JSON.stringify(users));
  } catch {
    // ignore storage quota errors
  }
}

export function saveSession(session: AuthSession, storage: StorageLike | null = getDefaultStorage()): void {
  if (!storage) return;
  try {
    storage.setItem(AUTH_SESSION_STORAGE_KEY, JSON.stringify(session));
  } catch {
    // ignore storage quota errors
  }
}

export function loadStoredSession(storage: StorageLike | null = getDefaultStorage()): AuthSession | null {
  if (!storage) return null;
  try {
    const raw = storage.getItem(AUTH_SESSION_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as AuthSession;
    if (!parsed?.token || !parsed?.user?.username || !parsed?.expiresAt) {
      return null;
    }
    if (new Date(parsed.expiresAt).getTime() <= Date.now()) {
      storage.removeItem(AUTH_SESSION_STORAGE_KEY);
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function clearStoredSession(storage: StorageLike | null = getDefaultStorage()): void {
  if (!storage) return;
  try {
    storage.removeItem(AUTH_SESSION_STORAGE_KEY);
  } catch {
    // ignore
  }
}

async function tryFastApiAuth<T>(
  apiUrl: string | undefined,
  path: string,
  options: RequestInit
): Promise<{ reached: boolean; ok: boolean; status: number; data?: T; detail?: string }> {
  if (!apiUrl || typeof fetch !== 'function') {
    return { reached: false, ok: false, status: 0 };
  }
  const cleanBase = apiUrl.trim().replace(/\/$/, '');
  if (!cleanBase) return { reached: false, ok: false, status: 0 };

  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 1600);
    const resp = await fetch(`${cleanBase}${path}`, {
      ...options,
      signal: ctrl.signal,
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers ?? {}),
      },
    });
    clearTimeout(timer);
    const json = (await resp.json().catch(() => ({}))) as Record<string, unknown>;
    if (!resp.ok) {
      return {
        reached: true,
        ok: false,
        status: resp.status,
        detail: String(json.detail ?? json.message ?? `Request failed with HTTP ${resp.status}.`),
      };
    }
    return { reached: true, ok: true, status: resp.status, data: json as T };
  } catch {
    return { reached: false, ok: false, status: 0 };
  }
}

export async function registerUserAccount(
  input: RegisterInput,
  options?: { apiUrl?: string; storage?: StorageLike | null }
): Promise<AuthActionResult> {
  const storage = options?.storage === undefined ? getDefaultStorage() : options.storage;
  const validation = validateRegistrationInput(input);
  if (!validation.valid) {
    const firstMsg = Object.values(validation.fieldErrors)[0] ?? 'Please fix the highlighted fields.';
    return {
      ok: false,
      message: firstMsg,
      fieldErrors: validation.fieldErrors,
    };
  }

  const { fullName, username, email } = validation.normalized;

  // Attempt FastAPI backend registration if running; fall back cleanly to local salted PBKDF2 store
  if (options?.apiUrl) {
    const apiRes = await tryFastApiAuth<{
      user: { user_id: string; full_name: string; username: string; email: string; created_at: string };
      message: string;
    }>(options.apiUrl, '/api/auth/register', {
      method: 'POST',
      body: JSON.stringify({
        full_name: fullName,
        username,
        email,
        password: input.password,
        confirm_password: input.confirmPassword,
      }),
    });

    if (apiRes.reached) {
      if (!apiRes.ok) {
        return {
          ok: false,
          message: apiRes.detail || 'Registration failed.',
        };
      }
      if (apiRes.data?.user) {
        const user: AuthUser = {
          userId: apiRes.data.user.user_id,
          fullName: apiRes.data.user.full_name,
          username: apiRes.data.user.username,
          email: apiRes.data.user.email,
          createdAt: apiRes.data.user.created_at,
        };
        return {
          ok: true,
          message: apiRes.data.message || `Account created for ${user.fullName}. Please sign in.`,
          user,
        };
      }
    }
  }

  const users = readStoredUsers(storage);
  if (users.some((u) => u.username.toLowerCase() === username)) {
    return {
      ok: false,
      message: `Username "${username}" is already registered. Please choose another.`,
      fieldErrors: { username: 'This username is already taken.' },
    };
  }
  if (users.some((u) => u.email.toLowerCase() === email)) {
    return {
      ok: false,
      message: `Email "${email}" is already registered. Try logging in instead.`,
      fieldErrors: { email: 'This email is already registered.' },
    };
  }

  const passwordHash = await hashPassword(input.password);
  const newRecord: StoredUserRecord = {
    userId: `USR-${randomHex(6).toUpperCase()}`,
    fullName,
    username,
    email,
    passwordHash,
    createdAt: new Date().toISOString(),
  };
  writeStoredUsers(storage, [...users, newRecord]);

  return {
    ok: true,
    message: `Account created successfully for ${fullName}. Please log in to continue.`,
    user: {
      userId: newRecord.userId,
      fullName: newRecord.fullName,
      username: newRecord.username,
      email: newRecord.email,
      createdAt: newRecord.createdAt,
    },
  };
}

export async function loginUserAccount(
  identifier: string,
  password: string,
  options?: { apiUrl?: string; storage?: StorageLike | null }
): Promise<AuthActionResult> {
  const storage = options?.storage === undefined ? getDefaultStorage() : options.storage;
  const cleanId = identifier.trim().toLowerCase();

  const fieldErrors: Partial<Record<'identifier' | 'password', string>> = {};
  if (!cleanId) fieldErrors.identifier = 'Username or email is required.';
  if (!password) fieldErrors.password = 'Password is required.';

  if (Object.keys(fieldErrors).length > 0) {
    return {
      ok: false,
      message: 'Please enter both username/email and password.',
      fieldErrors,
    };
  }

  if (options?.apiUrl) {
    const apiRes = await tryFastApiAuth<{
      access_token: string;
      user: { user_id: string; full_name: string; username: string; email: string; created_at: string };
      message: string;
    }>(options.apiUrl, '/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ identifier: cleanId, password }),
    });

    if (apiRes.reached) {
      if (!apiRes.ok) {
        return {
          ok: false,
          message: apiRes.detail || 'Invalid username/email or password.',
        };
      }
      if (apiRes.data?.access_token && apiRes.data?.user) {
        const user: AuthUser = {
          userId: apiRes.data.user.user_id,
          fullName: apiRes.data.user.full_name,
          username: apiRes.data.user.username,
          email: apiRes.data.user.email,
          createdAt: apiRes.data.user.created_at,
        };
        const session: AuthSession = {
          token: apiRes.data.access_token,
          user,
          authSource: 'api',
          expiresAt: new Date(Date.now() + SESSION_TTL_MS).toISOString(),
        };
        saveSession(session, storage);
        return {
          ok: true,
          message: apiRes.data.message || `Welcome back, ${user.fullName}.`,
          user,
          session,
        };
      }
    }
  }

  const users = readStoredUsers(storage);
  const found = users.find(
    (u) => u.username.toLowerCase() === cleanId || u.email.toLowerCase() === cleanId
  );

  if (!found) {
    return {
      ok: false,
      message: 'Invalid username/email or password.',
    };
  }

  const validPassword = await verifyPassword(password, found.passwordHash);
  if (!validPassword) {
    return {
      ok: false,
      message: 'Invalid username/email or password.',
    };
  }

  const user: AuthUser = {
    userId: found.userId,
    fullName: found.fullName,
    username: found.username,
    email: found.email,
    createdAt: found.createdAt,
  };
  const session: AuthSession = {
    token: `sp_local_${randomHex(24)}`,
    user,
    authSource: 'local',
    expiresAt: new Date(Date.now() + SESSION_TTL_MS).toISOString(),
  };
  saveSession(session, storage);

  return {
    ok: true,
    message: `Welcome back, ${user.fullName}.`,
    user,
    session,
  };
}

export async function restoreAuthSession(options?: {
  apiUrl?: string;
  storage?: StorageLike | null;
}): Promise<AuthSession | null> {
  const storage = options?.storage === undefined ? getDefaultStorage() : options.storage;
  const session = loadStoredSession(storage);
  if (!session) return null;

  if (session.authSource === 'api' && options?.apiUrl) {
    const check = await tryFastApiAuth<{
      user_id: string;
      full_name: string;
      username: string;
      email: string;
      created_at: string;
    }>(options.apiUrl, '/api/auth/me', {
      method: 'GET',
      headers: { Authorization: `Bearer ${session.token}` },
    });
    if (check.reached && !check.ok) {
      clearStoredSession(storage);
      return null;
    }
  }

  return session;
}

export async function logoutUserAccount(options?: {
  apiUrl?: string;
  storage?: StorageLike | null;
}): Promise<void> {
  const storage = options?.storage === undefined ? getDefaultStorage() : options.storage;
  const current = loadStoredSession(storage);
  if (current?.authSource === 'api' && options?.apiUrl) {
    await tryFastApiAuth(options.apiUrl, '/api/auth/logout', {
      method: 'POST',
      headers: { Authorization: `Bearer ${current.token}` },
    });
  }
  clearStoredSession(storage);
}

export function evaluateRouteAccess(
  requestedRoute: PublicRoute,
  session: AuthSession | null
): { allowedRoute: PublicRoute; notice?: string } {
  if (requestedRoute === 'app' && !session) {
    return {
      allowedRoute: 'login',
      notice: 'Please log in to access the SmartPrompt AI Dashboard.',
    };
  }
  return { allowedRoute: requestedRoute };
}

export function updateUserProfile(
  userId: string,
  updates: { fullName: string; email: string },
  options?: { storage?: StorageLike | null }
): { ok: boolean; message: string; session?: AuthSession } {
  const storage = options?.storage === undefined ? getDefaultStorage() : options.storage;
  const fullName = updates.fullName.trim();
  const email = updates.email.trim().toLowerCase();

  if (fullName.length < 2) {
    return { ok: false, message: 'Full name must be at least 2 characters.' };
  }
  if (!EMAIL_REGEX.test(email)) {
    return { ok: false, message: 'Please enter a valid email address.' };
  }

  const users = readStoredUsers(storage);
  if (users.some((u) => u.userId !== userId && u.email.toLowerCase() === email)) {
    return { ok: false, message: 'This email address is already used by another account.' };
  }

  const idx = users.findIndex((u) => u.userId === userId);
  if (idx >= 0) {
    users[idx] = { ...users[idx], fullName, email };
    writeStoredUsers(storage, users);
  }

  const currentSession = loadStoredSession(storage);
  if (currentSession && currentSession.user.userId === userId) {
    const updatedSession: AuthSession = {
      ...currentSession,
      user: {
        ...currentSession.user,
        fullName,
        email,
      },
    };
    saveSession(updatedSession, storage);
    return { ok: true, message: 'Profile information updated.', session: updatedSession };
  }

  return { ok: true, message: 'Profile information updated.' };
}

export async function changeUserPassword(
  userId: string,
  currentPassword: string,
  newPassword: string,
  confirmNewPassword: string,
  options?: { storage?: StorageLike | null }
): Promise<{ ok: boolean; message: string }> {
  const storage = options?.storage === undefined ? getDefaultStorage() : options.storage;

  if (!currentPassword || !newPassword || !confirmNewPassword) {
    return { ok: false, message: 'Please fill in all password fields.' };
  }
  if (newPassword.length < 8 || !/[A-Za-z]/.test(newPassword) || !/\d/.test(newPassword)) {
    return {
      ok: false,
      message: 'New password must be at least 8 characters and include a letter and a number.',
    };
  }
  if (newPassword !== confirmNewPassword) {
    return { ok: false, message: 'New password and confirmation do not match.' };
  }

  const users = readStoredUsers(storage);
  const idx = users.findIndex((u) => u.userId === userId);
  if (idx < 0) {
    return {
      ok: false,
      message: 'Password changes are managed by the connected authentication server for this session.',
    };
  }

  const validCurrent = await verifyPassword(currentPassword, users[idx].passwordHash);
  if (!validCurrent) {
    return { ok: false, message: 'Current password is incorrect.' };
  }

  const newHash = await hashPassword(newPassword);
  users[idx] = { ...users[idx], passwordHash: newHash };
  writeStoredUsers(storage, users);

  return { ok: true, message: 'Password updated successfully.' };
}
