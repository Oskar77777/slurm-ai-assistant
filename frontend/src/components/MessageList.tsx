import React, { useEffect, useRef } from 'react';
import Message from './Message';
import { Message as MessageType } from '../services/api';

interface MessageListProps {
  messages: MessageType[];
  isLoading: boolean;
}

const MessageList: React.FC<MessageListProps> = ({ messages, isLoading }) => {
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  if (messages.length === 0 && !isLoading) {
    return (
      <div className="message-list">
        <div className="welcome-message">
          <h2>Welcome to eX3 Cluster Assistant</h2>
          <p>I can help you with:</p>
          <ul>
            <li>Writing SLURM batch scripts</li>
            <li>Checking cluster resource availability</li>
            <li>Optimizing job submissions</li>
            <li>Troubleshooting HPC issues</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="message-list" ref={listRef}>
      {messages.map((message, index) => (
        <Message key={index} role={message.role} content={message.content} />
      ))}
      {isLoading && (
        <div className="message assistant">
          <div className="loading">
            <div className="loading-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageList;
