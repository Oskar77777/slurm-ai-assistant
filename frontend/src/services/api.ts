export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatResponse {
  response: string;
}

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export async function sendMessage(messages: Message[]): Promise<string> {
  const response = await fetch(`${API_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ messages }),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  const data: ChatResponse = await response.json();
  return data.response;
}
