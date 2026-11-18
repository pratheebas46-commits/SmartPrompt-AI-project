import React from 'react';

interface ErrorBoundaryState {
  hasError: boolean;
  errorMessage: string;
}

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, errorMessage: '' };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      errorMessage: error?.message || 'An unexpected rendering error occurred.',
    };
  }

  override render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center p-6">
          <div className="max-w-lg w-full bg-white border border-red-200 rounded-lg p-6 space-y-4">
            <h1 className="text-lg font-semibold text-red-700">
              SmartPrompt AI Encountered a Runtime Error
            </h1>
            <p className="text-sm text-slate-700">
              The application caught an unexpected error instead of rendering a blank screen:
            </p>
            <pre className="text-xs font-mono bg-red-50 border border-red-200 rounded p-3 text-red-900 overflow-x-auto whitespace-pre-wrap">
              {this.state.errorMessage}
            </pre>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => {
                  try {
                    localStorage.removeItem('smartprompt_ai_preview_history_v1');
                  } catch {
                    // ignore
                  }
                  window.location.reload();
                }}
                className="px-4 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 cursor-pointer"
              >
                Reset Local Cache &amp; Reload
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
