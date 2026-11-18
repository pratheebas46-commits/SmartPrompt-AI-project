import React, { useState } from 'react';
import {
  AlertCircle,
  Bell,
  CheckCircle2,
  Eye,
  EyeOff,
  FileSpreadsheet,
  Info,
  LogOut,
  Palette,
  Shield,
  User,
} from 'lucide-react';
import {
  changeUserPassword,
  updateUserProfile,
  type AuthSession,
} from '../engine/authService';
import type { UserPreferences } from '../engine/userPreferences';

interface SettingsViewProps {
  session: AuthSession;
  darkMode: boolean;
  preferences: UserPreferences;
  onUpdateDarkMode: (dark: boolean) => void;
  onUpdatePreferences: (next: UserPreferences) => void;
  onSessionUpdated: (next: AuthSession) => void;
  onLogout: () => void;
}

export function SettingsView({
  session,
  darkMode,
  preferences,
  onUpdateDarkMode,
  onUpdatePreferences,
  onSessionUpdated,
  onLogout,
}: SettingsViewProps) {
  // 1. Profile Settings state
  const [fullName, setFullName] = useState(session.user.fullName);
  const [email, setEmail] = useState(session.user.email);
  const [profileStatus, setProfileStatus] = useState<{ ok: boolean; message: string } | null>(null);

  // 5. Privacy & Security password change state
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmNewPassword, setConfirmNewPassword] = useState('');
  const [showCurrentPw, setShowCurrentPw] = useState(false);
  const [showNewPw, setShowNewPw] = useState(false);
  const [passwordStatus, setPasswordStatus] = useState<{ ok: boolean; message: string } | null>(null);
  const [updatingPassword, setUpdatingPassword] = useState(false);

  const [prefsSavedNotice, setPrefsSavedNotice] = useState<string | null>(null);

  const handleSaveProfile = (e: React.FormEvent) => {
    e.preventDefault();
    const res = updateUserProfile(session.user.userId, { fullName, email });
    setProfileStatus({ ok: res.ok, message: res.message });
    if (res.ok && res.session) {
      onSessionUpdated(res.session);
    }
  };

  const updatePref = <K extends keyof UserPreferences>(key: K, value: UserPreferences[K], label: string) => {
    const next = { ...preferences, [key]: value };
    onUpdatePreferences(next);
    setPrefsSavedNotice(`${label} updated.`);
    setTimeout(() => setPrefsSavedNotice(null), 2400);
  };

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    setPasswordStatus(null);
    setUpdatingPassword(true);
    try {
      const res = await changeUserPassword(
        session.user.userId,
        currentPassword,
        newPassword,
        confirmNewPassword
      );
      setPasswordStatus(res);
      if (res.ok) {
        setCurrentPassword('');
        setNewPassword('');
        setConfirmNewPassword('');
      }
    } finally {
      setUpdatingPassword(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-white">Settings</h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
            Manage your account profile, appearance, notifications, data preferences, and security.
          </p>
        </div>
        {prefsSavedNotice && (
          <div className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-700 dark:text-emerald-400">
            <CheckCircle2 className="h-3.5 w-3.5" />
            <span>{prefsSavedNotice}</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* 1. Profile Settings */}
        <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
          <div className="flex items-center gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
            <User className="h-4 w-4 text-blue-900 dark:text-indigo-400" />
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Profile Settings</h2>
          </div>

          <form onSubmit={handleSaveProfile} className="space-y-3.5">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="space-y-1">
                <label htmlFor="settings-fullname" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                  Full Name
                </label>
                <input
                  id="settings-fullname"
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="w-full px-3 py-1.5 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>
              <div className="space-y-1">
                <label className="block text-xs font-medium text-slate-700 dark:text-slate-300">Username</label>
                <input
                  type="text"
                  disabled
                  value={`@${session.user.username}`}
                  className="w-full px-3 py-1.5 text-sm border border-slate-200 dark:border-slate-800 rounded-md bg-slate-50 dark:bg-slate-800/60 text-slate-500 dark:text-slate-400"
                />
              </div>
            </div>

            <div className="space-y-1">
              <label htmlFor="settings-email" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                Email Address
              </label>
              <input
                id="settings-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3 py-1.5 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            <div className="text-xs text-slate-500 dark:text-slate-400 flex items-center justify-between pt-1">
              <span>Account ID: <span className="font-mono">{session.user.userId}</span></span>
              <span>Member since {new Date(session.user.createdAt).toLocaleDateString()}</span>
            </div>

            {profileStatus && (
              <div
                className={`flex items-center gap-2 p-2.5 rounded-md border text-xs ${
                  profileStatus.ok
                    ? 'border-emerald-200 dark:border-emerald-900 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300'
                    : 'border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/30 text-red-700 dark:text-red-300'
                }`}
              >
                {profileStatus.ok ? <CheckCircle2 className="h-3.5 w-3.5 shrink-0" /> : <AlertCircle className="h-3.5 w-3.5 shrink-0" />}
                <span>{profileStatus.message}</span>
              </div>
            )}

            <button
              type="submit"
              className="px-4 py-2 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-xs sm:text-sm font-medium hover:bg-blue-950 dark:hover:bg-indigo-700 cursor-pointer"
            >
              Save Profile
            </button>
          </form>
        </section>

        {/* 2. Appearance */}
        <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
          <div className="flex items-center gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
            <Palette className="h-4 w-4 text-blue-900 dark:text-indigo-400" />
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Appearance</h2>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <div className="text-xs font-medium text-slate-700 dark:text-slate-300">Color Theme</div>
              <div className="grid grid-cols-2 gap-2.5">
                <button
                  type="button"
                  onClick={() => {
                    onUpdateDarkMode(false);
                    setPrefsSavedNotice('Light theme enabled.');
                    setTimeout(() => setPrefsSavedNotice(null), 2000);
                  }}
                  className={`py-2.5 px-3 rounded-md border text-xs sm:text-sm font-medium transition-colors cursor-pointer ${
                    !darkMode
                      ? 'border-indigo-600 bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300'
                      : 'border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800'
                  }`}
                >
                  Light Mode
                </button>
                <button
                  type="button"
                  onClick={() => {
                    onUpdateDarkMode(true);
                    setPrefsSavedNotice('Dark theme enabled.');
                    setTimeout(() => setPrefsSavedNotice(null), 2000);
                  }}
                  className={`py-2.5 px-3 rounded-md border text-xs sm:text-sm font-medium transition-colors cursor-pointer ${
                    darkMode
                      ? 'border-indigo-600 bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300'
                      : 'border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800'
                  }`}
                >
                  Dark Mode
                </button>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between gap-4">
              <div>
                <div className="text-xs font-medium text-slate-800 dark:text-slate-200">Compact Table Density</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  Show more rows at once in data workspace and preview tables.
                </div>
              </div>
              <input
                id="pref-compact-tables"
                type="checkbox"
                checked={preferences.compactTables}
                onChange={(e) => updatePref('compactTables', e.target.checked, 'Table density')}
                className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
              />
            </div>
          </div>
        </section>

        {/* 3. Notifications */}
        <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
          <div className="flex items-center gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
            <Bell className="h-4 w-4 text-blue-900 dark:text-indigo-400" />
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Notifications</h2>
          </div>

          <div className="space-y-4">
            <label className="flex items-start justify-between gap-4 cursor-pointer">
              <div>
                <div className="text-xs font-medium text-slate-800 dark:text-slate-200">
                  Analysis Completion Notices
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  Show an in-app confirmation banner whenever a dataset analysis pipeline completes.
                </div>
              </div>
              <input
                type="checkbox"
                checked={preferences.notifyOnAnalysisComplete}
                onChange={(e) =>
                  updatePref('notifyOnAnalysisComplete', e.target.checked, 'Analysis completion notices')
                }
                className="h-4 w-4 mt-0.5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
              />
            </label>

            <label className="pt-3 border-t border-slate-100 dark:border-slate-800 flex items-start justify-between gap-4 cursor-pointer">
              <div>
                <div className="text-xs font-medium text-slate-800 dark:text-slate-200">
                  Report Export Confirmations
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  Display a confirmation banner after downloading Excel or CSV analytical reports.
                </div>
              </div>
              <input
                type="checkbox"
                checked={preferences.notifyOnReportExport}
                onChange={(e) => updatePref('notifyOnReportExport', e.target.checked, 'Report export notices')}
                className="h-4 w-4 mt-0.5 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
              />
            </label>
          </div>
        </section>

        {/* 4. Data Preferences */}
        <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
          <div className="flex items-center gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
            <FileSpreadsheet className="h-4 w-4 text-blue-900 dark:text-indigo-400" />
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Data Preferences</h2>
          </div>

          <div className="space-y-3.5">
            <div className="space-y-1.5">
              <label htmlFor="pref-export-format" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                Default Export Format
              </label>
              <select
                id="pref-export-format"
                value={preferences.defaultExportFormat}
                onChange={(e) =>
                  updatePref('defaultExportFormat', e.target.value as 'xlsx' | 'csv', 'Default export format')
                }
                className="w-full px-3 py-1.5 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
              >
                <option value="xlsx">Excel Workbook (.xlsx)</option>
                <option value="csv">Comma-Separated Values (.csv)</option>
              </select>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="space-y-1.5">
                <label htmlFor="pref-num-clean" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                  Default Numeric Imputation
                </label>
                <select
                  id="pref-num-clean"
                  value={preferences.defaultNumericCleaning}
                  onChange={(e) =>
                    updatePref(
                      'defaultNumericCleaning',
                      e.target.value as 'median' | 'mean' | 'zero',
                      'Numeric cleaning preference'
                    )
                  }
                  className="w-full px-3 py-1.5 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                >
                  <option value="median">Median</option>
                  <option value="mean">Mean</option>
                  <option value="zero">Fill with Zero</option>
                </select>
              </div>

              <div className="space-y-1.5">
                <label htmlFor="pref-cat-clean" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                  Default Category Imputation
                </label>
                <select
                  id="pref-cat-clean"
                  value={preferences.defaultCategoricalCleaning}
                  onChange={(e) =>
                    updatePref(
                      'defaultCategoricalCleaning',
                      e.target.value as 'mode' | 'unknown',
                      'Categorical cleaning preference'
                    )
                  }
                  className="w-full px-3 py-1.5 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                >
                  <option value="mode">Most Frequent (Mode)</option>
                  <option value="unknown">Fill as &quot;Unknown&quot;</option>
                </select>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Privacy & Security */}
        <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
          <div className="flex items-center gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
            <Shield className="h-4 w-4 text-blue-900 dark:text-indigo-400" />
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Privacy & Security</h2>
          </div>

          <form onSubmit={handleChangePassword} className="space-y-3">
            <div className="space-y-1">
              <label htmlFor="sec-cur-pw" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                Current Password
              </label>
              <div className="relative">
                <input
                  id="sec-cur-pw"
                  type={showCurrentPw ? 'text' : 'password'}
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  className="w-full pl-3 pr-9 py-1.5 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                />
                <button
                  type="button"
                  onClick={() => setShowCurrentPw((s) => !s)}
                  aria-label={showCurrentPw ? 'Hide current password' : 'Show current password'}
                  className="absolute inset-y-0 right-0 px-2.5 flex items-center text-slate-400 hover:text-slate-600 cursor-pointer"
                >
                  {showCurrentPw ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="space-y-1">
                <label htmlFor="sec-new-pw" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                  New Password
                </label>
                <div className="relative">
                  <input
                    id="sec-new-pw"
                    type={showNewPw ? 'text' : 'password'}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    placeholder="Min 8 chars, letters + numbers"
                    className="w-full pl-3 pr-9 py-1.5 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                  />
                  <button
                    type="button"
                    onClick={() => setShowNewPw((s) => !s)}
                    aria-label={showNewPw ? 'Hide new password' : 'Show new password'}
                    className="absolute inset-y-0 right-0 px-2.5 flex items-center text-slate-400 hover:text-slate-600 cursor-pointer"
                  >
                    {showNewPw ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                  </button>
                </div>
              </div>

              <div className="space-y-1">
                <label htmlFor="sec-conf-pw" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                  Confirm New Password
                </label>
                <input
                  id="sec-conf-pw"
                  type={showNewPw ? 'text' : 'password'}
                  value={confirmNewPassword}
                  onChange={(e) => setConfirmNewPassword(e.target.value)}
                  className="w-full px-3 py-1.5 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white"
                />
              </div>
            </div>

            {passwordStatus && (
              <div
                className={`flex items-center gap-2 p-2.5 rounded-md border text-xs ${
                  passwordStatus.ok
                    ? 'border-emerald-200 dark:border-emerald-900 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300'
                    : 'border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/30 text-red-700 dark:text-red-300'
                }`}
              >
                {passwordStatus.ok ? <CheckCircle2 className="h-3.5 w-3.5 shrink-0" /> : <AlertCircle className="h-3.5 w-3.5 shrink-0" />}
                <span>{passwordStatus.message}</span>
              </div>
            )}

            <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
              <button
                type="submit"
                disabled={updatingPassword}
                className="px-4 py-2 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-xs sm:text-sm font-medium hover:bg-blue-950 dark:hover:bg-indigo-700 disabled:opacity-60 cursor-pointer"
              >
                {updatingPassword ? 'Updating...' : 'Update Password'}
              </button>

              <button
                type="button"
                onClick={onLogout}
                className="inline-flex items-center gap-1.5 px-3 py-2 rounded-md border border-slate-200 dark:border-slate-700 text-xs font-medium text-red-700 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30 cursor-pointer"
              >
                <LogOut className="h-3.5 w-3.5" />
                <span>Sign Out</span>
              </button>
            </div>
          </form>
        </section>

        {/* 6. About SmartPrompt AI */}
        <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
          <div className="flex items-center gap-2 border-b border-slate-100 dark:border-slate-800 pb-3">
            <Info className="h-4 w-4 text-blue-900 dark:text-indigo-400" />
            <h2 className="text-sm font-semibold text-slate-900 dark:text-white">About SmartPrompt AI</h2>
          </div>

          <div className="space-y-3 text-xs sm:text-sm text-slate-600 dark:text-slate-300">
            <div className="flex items-center justify-between">
              <span className="text-slate-500 dark:text-slate-400">Application</span>
              <span className="font-medium text-slate-900 dark:text-white">SmartPrompt AI</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500 dark:text-slate-400">Version</span>
              <span className="font-mono text-slate-900 dark:text-white">1.0.0</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-500 dark:text-slate-400">Support Contact</span>
              <span className="font-mono text-indigo-700 dark:text-indigo-400">support@smartprompt.ai</span>
            </div>
            <p className="pt-2 border-t border-slate-100 dark:border-slate-800 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
              SmartPrompt AI helps analysts upload datasets or documents, clean data, optimize analytical prompts,
              detect anomalies, validate metrics, and export verified reports.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
