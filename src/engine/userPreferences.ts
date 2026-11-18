import type { StorageLike } from './authService';

export interface UserPreferences {
  compactTables: boolean;
  notifyOnAnalysisComplete: boolean;
  notifyOnReportExport: boolean;
  defaultExportFormat: 'xlsx' | 'csv';
  defaultNumericCleaning: 'median' | 'mean' | 'zero';
  defaultCategoricalCleaning: 'mode' | 'unknown';
}

export const USER_PREFS_STORAGE_KEY = 'smartprompt_ai_user_prefs_v1';

export const DEFAULT_USER_PREFERENCES: UserPreferences = {
  compactTables: false,
  notifyOnAnalysisComplete: true,
  notifyOnReportExport: true,
  defaultExportFormat: 'xlsx',
  defaultNumericCleaning: 'median',
  defaultCategoricalCleaning: 'mode',
};

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

export function loadUserPreferences(storage: StorageLike | null = getDefaultStorage()): UserPreferences {
  if (!storage) return { ...DEFAULT_USER_PREFERENCES };
  try {
    const raw = storage.getItem(USER_PREFS_STORAGE_KEY);
    if (!raw) return { ...DEFAULT_USER_PREFERENCES };
    const parsed = JSON.parse(raw) as Partial<UserPreferences>;
    return {
      compactTables: Boolean(parsed.compactTables),
      notifyOnAnalysisComplete:
        parsed.notifyOnAnalysisComplete === undefined ? true : Boolean(parsed.notifyOnAnalysisComplete),
      notifyOnReportExport:
        parsed.notifyOnReportExport === undefined ? true : Boolean(parsed.notifyOnReportExport),
      defaultExportFormat: parsed.defaultExportFormat === 'csv' ? 'csv' : 'xlsx',
      defaultNumericCleaning:
        parsed.defaultNumericCleaning === 'mean' || parsed.defaultNumericCleaning === 'zero'
          ? parsed.defaultNumericCleaning
          : 'median',
      defaultCategoricalCleaning:
        parsed.defaultCategoricalCleaning === 'unknown' ||
        (parsed.defaultCategoricalCleaning as string | undefined) === 'constant'
          ? 'unknown'
          : 'mode',
    };
  } catch {
    return { ...DEFAULT_USER_PREFERENCES };
  }
}

export function saveUserPreferences(
  prefs: UserPreferences,
  storage: StorageLike | null = getDefaultStorage()
): void {
  if (!storage) return;
  try {
    storage.setItem(USER_PREFS_STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    // ignore storage quota errors
  }
}
