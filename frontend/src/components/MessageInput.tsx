import React, { useState, KeyboardEvent, ChangeEvent } from 'react';

interface MessageInputProps {
  onSend: (message: string) => void;
  disabled: boolean;
}

const MessageInput: React.FC<MessageInputProps> = ({ onSend, disabled }) => {
  const [input, setInput] = useState('');

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (trimmed && !disabled) {
      onSend(trimmed);
      setInput('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  return (
    <div className="message-input">
      <textarea
        value={input}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder="Ask about SLURM, cluster resources, or HPC workflows..."
        disabled={disabled}
        rows={1}
      />
      <button onClick={handleSubmit} disabled={disabled || !input.trim()}>
        Send
      </button>
    </div>
  );
};

export default MessageInput;
