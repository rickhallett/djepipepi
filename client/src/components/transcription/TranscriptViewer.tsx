import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';

interface TranscriptViewerProps {
  content: string;
}

interface CodeProps extends React.HTMLAttributes<HTMLElement> {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
}

export function TranscriptViewer({ content }: TranscriptViewerProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Transcript</CardTitle>
        <CardDescription>
          Completed transcription result
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="prose prose-sm max-w-none dark:prose-invert">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              h1: (props) => <h1 className="text-2xl font-bold mt-6 mb-4" {...props} />,
              h2: (props) => <h2 className="text-xl font-bold mt-5 mb-3" {...props} />,
              h3: (props) => <h3 className="text-lg font-bold mt-4 mb-2" {...props} />,
              p: (props) => <p className="mb-4" {...props} />,
              ul: (props) => <ul className="list-disc ml-6 mb-4" {...props} />,
              ol: (props) => <ol className="list-decimal ml-6 mb-4" {...props} />,
              li: (props) => <li className="mb-1" {...props} />,
              blockquote: (props) => (
                <blockquote className="border-l-4 border-gray-200 pl-4 italic my-4" {...props} />
              ),
              code: ({ inline, className, children, ...props }: CodeProps) => {
                return inline ? (
                  <code className="bg-gray-100 rounded px-1 py-0.5 text-sm" {...props}>
                    {children}
                  </code>
                ) : (
                  <pre className="bg-gray-100 rounded p-4 overflow-auto text-sm my-4">
                    <code className={className} {...props}>
                      {children}
                    </code>
                  </pre>
                );
              }
            }}
          >
            {content}
          </ReactMarkdown>
        </div>
      </CardContent>
    </Card>
  );
} 