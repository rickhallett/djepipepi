import React from 'react';
import { Layout } from './components/layout/Layout';
import { Dashboard } from './components/transcription/Dashboard';
import './index.css';
import './App.css'

function App() {
  return (
    <Layout>
      <Dashboard />
    </Layout>
  );
}

export default App;
