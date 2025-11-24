import { useEffect, useRef, useState } from 'react';

export default function Mermaid({ chart, figure, alt }) {
  const mermaidRef = useRef(null);
  const [isClient, setIsClient] = useState(false);
  const [uniqueId] = useState(() => `mermaid-${Math.random().toString(36).substr(2, 9)}`);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    if (!isClient) return;

    const initializeMermaid = async () => {
      // Dynamic import to avoid SSR issues
      const mermaid = (await import('mermaid')).default;
      
      mermaid.initialize({
        startOnLoad: false,
        theme: 'default',
        securityLevel: 'loose',
        fontFamily: 'inherit'
      });

      try {
        const { svg } = await mermaid.render(uniqueId, chart);
        if (mermaidRef.current) {
          mermaidRef.current.innerHTML = svg;
        }
      } catch (error) {
        console.error('Error rendering Mermaid diagram:', error);
        if (mermaidRef.current) {
          mermaidRef.current.innerHTML = `
            <div style="border: 1px solid #e5e7eb; border-radius: 0.375rem; padding: 1rem; background: #f9fafb; color: #6b7280;">
              <p style="font-weight: 600; margin-bottom: 0.5rem;">Error rendering diagram:</p>
              <pre style="white-space: pre-wrap; font-size: 0.875rem;">${error.message}</pre>
              <details style="margin-top: 0.5rem;">
                <summary style="cursor: pointer; font-size: 0.875rem;">Original diagram code</summary>
                <pre style="white-space: pre-wrap; margin-top: 0.25rem; font-size: 0.75rem;">${chart}</pre>
              </details>
            </div>
          `;
        }
      }
    };

    initializeMermaid();
  }, [chart, isClient]);

  if (!isClient) {
    const loadingContent = (
      <div className="my-6 text-center" style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100px',
        background: '#f9fafb',
        border: '1px solid #e5e7eb',
        borderRadius: '0.375rem',
        color: '#6b7280'
      }}>
        Loading diagram...
      </div>
    );

    if (figure) {
      return (
        <figure className="my-6">
          {loadingContent}
          {alt && <figcaption className="mt-2 text-sm text-gray-600 dark:text-gray-400 text-center">{alt}</figcaption>}
        </figure>
      );
    }

    return loadingContent;
  }

  const diagramContent = (
    <div
      ref={mermaidRef}
      className="my-6 text-center"
      style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100px'
      }}
      role="img"
      aria-label={alt || 'Mermaid diagram'}
    />
  );

  if (figure) {
    return (
      <figure className="my-6">
        {diagramContent}
        {alt && <figcaption className="mt-2 text-sm text-gray-600 dark:text-gray-400 text-center">{alt}</figcaption>}
      </figure>
    );
  }

  return diagramContent;
}