import React from 'react';
import { Header } from './Header';

interface LayoutProps {
  children: React.ReactNode;
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen flex flex-col bg-background">
      <Header />
      <main className="flex-1 container mx-auto p-6">
        {children}
      </main>
      <footer className="py-4 px-6 text-center text-sm text-muted-foreground">
        <p>© {new Date().getFullYear()} Audio Transcription App</p>
      </footer>
    </div>
  );
} 