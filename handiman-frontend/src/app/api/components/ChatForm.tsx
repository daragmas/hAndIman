'use client';

import { useState, useEffect, useRef } from 'react';
interface Message {
  id: string;
  role: 'user' | 'ai';
  content: string;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage: Message = { id: Date.now().toString(), role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userMessage.content }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      const aiMessage: Message = { id: Date.now().toString(), role: 'ai', content: data.message };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error('Failed to fetch from Flask API:', error);
      const errorMessage: Message = { id: Date.now().toString(), role: 'ai', content: 'Sorry, something went wrong.' };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  return (
    <div className="flex h-full flex-col">
      {/* Message Area */}
      <div className="flex-1 overflow-y-auto p-4" ref={scrollRef}>
        {messages.length > 0 ? (
          messages.map((m) => (
            <div key={m.id} className={`mb-4 ${m.role === 'user' ? 'text-right' : 'text-left'}`}>
              <p className="text-sm font-semibold">{m.role === 'user' ? 'User' : 'AI'}</p>
              <div className={`inline-block rounded-lg px-4 py-2 ${m.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'}`}>
                {m.content}
              </div>
            </div>
          ))
        ) : (
          <div className="flex h-full items-center justify-center text-gray-500">
            Start a conversation...
          </div>
        )}
      </div>

      {/* Input Form */}
      <form onSubmit={handleSubmit} className="border-t p-4">
        <div className="flex space-x-2">
          <input
            className="flex-1 rounded-lg border p-2"
            placeholder="Say something..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button type="submit" className="rounded-lg bg-blue-500 px-4 py-2 text-white">
            Send
          </button>
        </div>
      </form>
    </div>
  );
}