import { useState } from 'react';
import Mermaid from '../components/Mermaid';

export default function TestMermaid() {
  const [showDiagram, setShowDiagram] = useState(false);

  return (
    <div style={{ padding: '2rem' }}>
      <h1>Mermaid Test Page</h1>
      <button 
        onClick={() => setShowDiagram(!showDiagram)}
        style={{ 
          padding: '0.5rem 1rem', 
          marginBottom: '1rem',
          backgroundColor: '#0070f3',
          color: 'white',
          border: 'none',
          borderRadius: '0.25rem',
          cursor: 'pointer'
        }}
      >
        {showDiagram ? 'Hide' : 'Show'} Mermaid Diagram
      </button>
      
      {showDiagram && (
        <Mermaid chart={`graph TD
    A[Start] --> B[Process]
    B --> C[End]
    
    style A fill:#e1f5fe
    style C fill:#c8e6c9`} />
      )}
    </div>
  );
}
