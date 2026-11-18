import React, { useState } from 'react';
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Eye,
  EyeOff,
  LayoutDashboard,
  Menu,
  Moon,
  Sun,
  UserPlus,
  X,
} from 'lucide-react';
import {
  loginUserAccount,
  registerUserAccount,
  type AuthSession,
  type PublicRoute,
  type RegisterInput,
} from '../engine/authService';

interface PublicHeaderProps {
  currentRoute: PublicRoute;
  session: AuthSession | null;
  darkMode: boolean;
  onToggleDarkMode: () => void;
  onNavigate: (route: PublicRoute, sectionId?: string) => void;
  onLogout: () => void;
}

function PublicNavbar({
  currentRoute,
  session,
  darkMode,
  onToggleDarkMode,
  onNavigate,
  onLogout,
}: PublicHeaderProps) {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const handleLinkClick = (route: PublicRoute, sectionId?: string) => {
    setMobileNavOpen(false);
    onNavigate(route, sectionId);
  };

  return (
    <header className="sticky top-0 z-30 bg-white/95 dark:bg-slate-900/95 backdrop-blur border-b border-slate-200 dark:border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
        <button
          type="button"
          onClick={() => handleLinkClick('home')}
          className="text-left cursor-pointer"
        >
          <div className="font-semibold text-base sm:text-lg tracking-tight text-slate-900 dark:text-white leading-tight">
            SmartPrompt AI
          </div>
          <div className="text-[11px] text-slate-500 dark:text-slate-400 hidden sm:block">
            Intelligent Data Analyst
          </div>
        </button>

        {/* Desktop Nav */}
        <nav className="hidden md:flex items-center gap-1" aria-label="Primary navigation">
          <button
            type="button"
            onClick={() => handleLinkClick('home', 'top')}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors cursor-pointer ${
              currentRoute === 'home'
                ? 'text-indigo-700 dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-950/60'
                : 'text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800'
            }`}
          >
            Home
          </button>
        </nav>

        <div className="hidden md:flex items-center gap-2">
          <button
            type="button"
            onClick={onToggleDarkMode}
            className="p-2 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 cursor-pointer"
            aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>

          {session ? (
            <>
              <button
                type="button"
                onClick={() => handleLinkClick('app')}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-sm font-medium hover:bg-blue-950 dark:hover:bg-indigo-700 cursor-pointer"
              >
                <LayoutDashboard className="h-4 w-4" />
                <span>Dashboard</span>
              </button>
              <button
                type="button"
                onClick={onLogout}
                className="px-3 py-1.5 rounded-md border border-slate-200 dark:border-slate-700 text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 cursor-pointer"
              >
                Logout
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                onClick={() => handleLinkClick('login')}
                className={`px-3.5 py-1.5 rounded-md border text-sm font-medium transition-colors cursor-pointer ${
                  currentRoute === 'login'
                    ? 'border-indigo-600 text-indigo-700 dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-950/50'
                    : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700'
                }`}
              >
                Login
              </button>
              <button
                type="button"
                onClick={() => handleLinkClick('register')}
                className="px-3.5 py-1.5 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-sm font-medium hover:bg-blue-950 dark:hover:bg-indigo-700 transition-colors cursor-pointer"
              >
                Register
              </button>
            </>
          )}
        </div>

        {/* Mobile menu button */}
        <div className="flex md:hidden items-center gap-2">
          <button
            type="button"
            onClick={onToggleDarkMode}
            className="p-2 rounded-md border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300"
            aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
          <button
            type="button"
            onClick={() => setMobileNavOpen((o) => !o)}
            className="p-2 rounded-md border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200"
            aria-label="Toggle navigation menu"
          >
            {mobileNavOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {mobileNavOpen && (
        <div className="md:hidden border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 px-4 py-3 space-y-1">
          <button
            type="button"
            onClick={() => handleLinkClick('home', 'top')}
            className="block w-full text-left px-3 py-2 rounded-md text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            Home
          </button>
          <div className="pt-2 border-t border-slate-100 dark:border-slate-800 flex gap-2">
            {session ? (
              <>
                <button
                  type="button"
                  onClick={() => handleLinkClick('app')}
                  className="flex-1 py-2 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-sm font-medium"
                >
                  Dashboard
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setMobileNavOpen(false);
                    onLogout();
                  }}
                  className="flex-1 py-2 rounded-md border border-slate-200 dark:border-slate-700 text-sm font-medium"
                >
                  Logout
                </button>
              </>
            ) : (
              <>
                <button
                  type="button"
                  onClick={() => handleLinkClick('login')}
                  className="flex-1 py-2 rounded-md border border-slate-200 dark:border-slate-700 text-sm font-medium"
                >
                  Login
                </button>
                <button
                  type="button"
                  onClick={() => handleLinkClick('register')}
                  className="flex-1 py-2 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-sm font-medium"
                >
                  Register
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </header>
  );
}

interface HomePageProps {
  session: AuthSession | null;
  darkMode: boolean;
  onToggleDarkMode: () => void;
  onNavigate: (route: PublicRoute, sectionId?: string) => void;
  onLogout: () => void;
}

export function HomePage({
  session,
  darkMode,
  onToggleDarkMode,
  onNavigate,
  onLogout,
}: HomePageProps) {
  return (
    <div className="min-h-screen flex flex-col bg-[#F4F5F7] dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <PublicNavbar
        currentRoute="home"
        session={session}
        darkMode={darkMode}
        onToggleDarkMode={onToggleDarkMode}
        onNavigate={onNavigate}
        onLogout={onLogout}
      />

      <main className="flex-1 flex items-center bg-white dark:bg-slate-900">
        {/* Hero Section */}
        <section id="top" className="w-full">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 py-20 lg:py-28 space-y-6">
            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-semibold tracking-tight text-slate-900 dark:text-white leading-[1.15]">
              SmartPrompt AI – Intelligent Data Analysis Made Simple
            </h1>
            <p className="text-base sm:text-lg text-slate-600 dark:text-slate-300 leading-relaxed max-w-2xl">
              Upload structured CSV or Excel datasets, automate data cleaning, transform ambiguous questions into
              structured analytical prompts, and generate verified insights backed by deterministic calculations.
            </p>

            <div className="flex flex-wrap items-center gap-3 pt-2">
              <button
                type="button"
                onClick={() => onNavigate(session ? 'app' : 'login')}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-sm font-medium hover:bg-blue-950 dark:hover:bg-indigo-700 transition-colors cursor-pointer"
              >
                <span>{session ? 'Open Dashboard' : 'Get Started'}</span>
                <ArrowRight className="h-4 w-4" />
              </button>
              {!session && (
                <button
                  type="button"
                  onClick={() => onNavigate('register')}
                  className="inline-flex items-center gap-2 px-5 py-2.5 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-100 text-sm font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors cursor-pointer"
                >
                  Create Account
                </button>
              )}
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-white dark:bg-slate-900 border-t border-slate-200 dark:border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500 dark:text-slate-400">
          <div className="font-medium text-slate-700 dark:text-slate-200">
            SmartPrompt AI
          </div>
          <div className="flex items-center gap-4">
            <button type="button" onClick={() => onNavigate('home', 'top')} className="hover:text-slate-800 dark:hover:text-slate-200 cursor-pointer">
              Home
            </button>
            <button type="button" onClick={() => onNavigate('login')} className="hover:text-slate-800 dark:hover:text-slate-200 cursor-pointer">
              Login
            </button>
            <button type="button" onClick={() => onNavigate('register')} className="hover:text-slate-800 dark:hover:text-slate-200 cursor-pointer">
              Register
            </button>
          </div>
          <div>© {new Date().getFullYear()} SmartPrompt AI. All rights reserved.</div>
        </div>
      </footer>
    </div>
  );
}

interface LoginPageProps {
  apiUrl: string;
  darkMode: boolean;
  notice?: string | null;
  prefilledIdentifier?: string;
  onToggleDarkMode: () => void;
  onNavigate: (route: PublicRoute, sectionId?: string) => void;
  onLoginSuccess: (session: AuthSession) => void;
}

export function LoginPage({
  apiUrl,
  darkMode,
  notice,
  prefilledIdentifier = '',
  onToggleDarkMode,
  onNavigate,
  onLoginSuccess,
}: LoginPageProps) {
  const [identifier, setIdentifier] = useState(prefilledIdentifier);
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);

    if (!identifier.trim() || !password) {
      setErrorMessage('Please enter both your username/email and password.');
      return;
    }

    setSubmitting(true);
    try {
      const res = await loginUserAccount(identifier, password, { apiUrl });
      if (!res.ok || !res.session) {
        setErrorMessage(res.message);
        return;
      }
      onLoginSuccess(res.session);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F4F5F7] dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <PublicNavbar
        currentRoute="login"
        session={null}
        darkMode={darkMode}
        onToggleDarkMode={onToggleDarkMode}
        onNavigate={onNavigate}
        onLogout={() => undefined}
      />

      <main className="flex-1 flex items-center justify-center px-4 py-12">
        <div className="w-full max-w-md bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-6 sm:p-8 shadow-[0_1px_2px_rgba(15,23,42,0.03)] space-y-6">
          <div className="space-y-1.5">
            <h1 className="text-xl font-semibold text-slate-900 dark:text-white tracking-tight">
              Sign in to SmartPrompt AI
            </h1>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Access your analytical workspace, prompt optimizer, and validation reports.
            </p>
          </div>

          {notice && (
            <div
              role="status"
              className="flex items-start gap-2.5 p-3 rounded-md border border-emerald-200 dark:border-emerald-900 bg-emerald-50 dark:bg-emerald-950/40 text-xs sm:text-sm text-emerald-800 dark:text-emerald-300"
            >
              <CheckCircle2 className="h-4 w-4 shrink-0 mt-0.5" />
              <span>{notice}</span>
            </div>
          )}

          {errorMessage && (
            <div
              role="alert"
              className="flex items-start gap-2.5 p-3 rounded-md border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/40 text-xs sm:text-sm text-red-700 dark:text-red-300"
            >
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <span>{errorMessage}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate className="space-y-4">
            <div className="space-y-1.5">
              <label htmlFor="login-identifier" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                Username or Email
              </label>
              <input
                id="login-identifier"
                type="text"
                autoComplete="username"
                value={identifier}
                onChange={(e) => setIdentifier(e.target.value)}
                placeholder="e.g. aarav_m or aarav@company.com"
                className="w-full px-3 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            </div>

            <div className="space-y-1.5">
              <label htmlFor="login-password" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                Password
              </label>
              <div className="relative">
                <input
                  id="login-password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="w-full pl-3 pr-10 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  className="absolute inset-y-0 right-0 px-2.5 flex items-center text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 cursor-pointer"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={submitting}
              className="w-full py-2.5 px-4 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-sm font-medium hover:bg-blue-950 dark:hover:bg-indigo-700 disabled:opacity-60 transition-colors cursor-pointer"
            >
              {submitting ? 'Signing in...' : 'Login'}
            </button>
          </form>

          <div className="pt-4 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-xs text-slate-600 dark:text-slate-400">
            <span>New to SmartPrompt AI?</span>
            <button
              type="button"
              onClick={() => onNavigate('register')}
              className="font-semibold text-indigo-700 dark:text-indigo-400 hover:underline cursor-pointer"
            >
              Create an account
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

interface RegisterPageProps {
  apiUrl: string;
  darkMode: boolean;
  onToggleDarkMode: () => void;
  onNavigate: (route: PublicRoute, sectionId?: string) => void;
  onRegisterSuccess: (username: string, message: string) => void;
}

export function RegisterPage({
  apiUrl,
  darkMode,
  onToggleDarkMode,
  onNavigate,
  onRegisterSuccess,
}: RegisterPageProps) {
  const [form, setForm] = useState<RegisterInput>({
    fullName: '',
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [fieldErrors, setFieldErrors] = useState<Partial<Record<keyof RegisterInput, string>>>({});
  const [submitting, setSubmitting] = useState(false);

  const updateField = (key: keyof RegisterInput, value: string) => {
    setForm((prev) => ({ ...prev, [key]: value }));
    if (fieldErrors[key]) {
      setFieldErrors((prev) => ({ ...prev, [key]: undefined }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    setFieldErrors({});
    setSubmitting(true);

    try {
      const res = await registerUserAccount(form, { apiUrl });
      if (!res.ok) {
        setErrorMessage(res.message);
        if (res.fieldErrors) setFieldErrors(res.fieldErrors);
        return;
      }
      onRegisterSuccess(res.user?.username ?? form.username.trim(), res.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#F4F5F7] dark:bg-slate-950 text-slate-900 dark:text-slate-100">
      <PublicNavbar
        currentRoute="register"
        session={null}
        darkMode={darkMode}
        onToggleDarkMode={onToggleDarkMode}
        onNavigate={onNavigate}
        onLogout={() => undefined}
      />

      <main className="flex-1 flex items-center justify-center px-4 py-10">
        <div className="w-full max-w-md bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-6 sm:p-8 shadow-[0_1px_2px_rgba(15,23,42,0.03)] space-y-5">
          <div className="space-y-1.5">
            <h1 className="text-xl font-semibold text-slate-900 dark:text-white tracking-tight flex items-center gap-2">
              <UserPlus className="h-5 w-5 text-blue-900 dark:text-indigo-400" />
              <span>Create Your Account</span>
            </h1>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Register to unlock dataset profiling, smart prompt optimization, and validated insights.
            </p>
          </div>

          {errorMessage && (
            <div
              role="alert"
              className="flex items-start gap-2.5 p-3 rounded-md border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/40 text-xs sm:text-sm text-red-700 dark:text-red-300"
            >
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <span>{errorMessage}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate className="space-y-3.5">
            <div className="space-y-1">
              <label htmlFor="reg-fullname" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                Full Name
              </label>
              <input
                id="reg-fullname"
                type="text"
                autoComplete="name"
                value={form.fullName}
                onChange={(e) => updateField('fullName', e.target.value)}
                placeholder="e.g. Aarav Mehta"
                className="w-full px-3 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              {fieldErrors.fullName && <p className="text-xs text-red-600 dark:text-red-400">{fieldErrors.fullName}</p>}
            </div>

            <div className="space-y-1">
              <label htmlFor="reg-username" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                Username
              </label>
              <input
                id="reg-username"
                type="text"
                autoComplete="username"
                value={form.username}
                onChange={(e) => updateField('username', e.target.value)}
                placeholder="e.g. aarav_mehta"
                className="w-full px-3 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              {fieldErrors.username && <p className="text-xs text-red-600 dark:text-red-400">{fieldErrors.username}</p>}
            </div>

            <div className="space-y-1">
              <label htmlFor="reg-email" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                Email Address
              </label>
              <input
                id="reg-email"
                type="email"
                autoComplete="email"
                value={form.email}
                onChange={(e) => updateField('email', e.target.value)}
                placeholder="e.g. aarav@company.com"
                className="w-full px-3 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              {fieldErrors.email && <p className="text-xs text-red-600 dark:text-red-400">{fieldErrors.email}</p>}
            </div>

            <div className="space-y-1">
              <label htmlFor="reg-password" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                Password
              </label>
              <div className="relative">
                <input
                  id="reg-password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  value={form.password}
                  onChange={(e) => updateField('password', e.target.value)}
                  placeholder="Min 8 chars, include letters & numbers"
                  className="w-full pl-3 pr-10 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((s) => !s)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  className="absolute inset-y-0 right-0 px-2.5 flex items-center text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 cursor-pointer"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {fieldErrors.password && <p className="text-xs text-red-600 dark:text-red-400">{fieldErrors.password}</p>}
            </div>

            <div className="space-y-1">
              <label htmlFor="reg-confirm" className="block text-xs font-medium text-slate-700 dark:text-slate-300">
                Confirm Password
              </label>
              <div className="relative">
                <input
                  id="reg-confirm"
                  type={showConfirm ? 'text' : 'password'}
                  autoComplete="new-password"
                  value={form.confirmPassword}
                  onChange={(e) => updateField('confirmPassword', e.target.value)}
                  placeholder="Re-enter your password"
                  className="w-full pl-3 pr-10 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirm((s) => !s)}
                  aria-label={showConfirm ? 'Hide confirm password' : 'Show confirm password'}
                  className="absolute inset-y-0 right-0 px-2.5 flex items-center text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 cursor-pointer"
                >
                  {showConfirm ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              {fieldErrors.confirmPassword && (
                <p className="text-xs text-red-600 dark:text-red-400">{fieldErrors.confirmPassword}</p>
              )}
            </div>

            <button
              type="submit"
              disabled={submitting}
              className="w-full py-2.5 px-4 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-sm font-medium hover:bg-blue-950 dark:hover:bg-indigo-700 disabled:opacity-60 transition-colors cursor-pointer"
            >
              {submitting ? 'Creating account...' : 'Register'}
            </button>
          </form>

          <div className="pt-4 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between text-xs text-slate-600 dark:text-slate-400">
            <span>Already have an account?</span>
            <button
              type="button"
              onClick={() => onNavigate('login')}
              className="font-semibold text-indigo-700 dark:text-indigo-400 hover:underline cursor-pointer"
            >
              Login
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
