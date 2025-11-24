// Utility functions for markdown, text formatting, and time formatting

// Function to parse markdown bold syntax
export const parseMarkdown = (text) => {
  const parts = [];
  let lastIndex = 0;
  const boldRegex = /\*\*(.+?)\*\*/g;
  let match;
  while ((match = boldRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push({ type: 'text', content: text.substring(lastIndex, match.index) });
    }
    parts.push({ type: 'bold', content: match[1] });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push({ type: 'text', content: text.substring(lastIndex) });
  }
  return parts.length > 0 ? parts : [{ type: 'text', content: text }];
};

// Function to detect and render URLs in text
export const renderTextWithLinks = (text, key) => {
  const markdownLinkRegex = /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g;
  const bareUrlRegex = /\((https?:\/\/[^)]+)\)/g;
  const parts = [];
  let lastIndex = 0;
  let match;
  const processedText = text.replace(markdownLinkRegex, (fullMatch, linkText, url) => {
    return `__MARKDOWN_LINK__${linkText}__URL__${url}__END__`;
  });
  const linkPlaceholderRegex = /__MARKDOWN_LINK__([^_]+)__URL__([^_]+)__END__/g;
  while ((match = linkPlaceholderRegex.exec(processedText)) !== null) {
    if (match.index > lastIndex) {
      const beforeText = processedText.substring(lastIndex, match.index);
      let cleanedBeforeText = beforeText.replace(bareUrlRegex, '');
      cleanedBeforeText = cleanedBeforeText.replace(/Link\s*:\s*$/i, '');
      if (cleanedBeforeText.trim()) {
        parts.push(<span key={`text-${key}-${lastIndex}`}>{cleanedBeforeText}</span>);
      }
    }
    parts.push(
      <a 
        key={`link-${key}-${match.index}`}
        href={match[2]} 
        target="_blank" 
        rel="noopener noreferrer"
        className="markdown-link"
      >
        {match[1]}
      </a>
    );
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < processedText.length) {
    const remainingText = processedText.substring(lastIndex);
    const cleanedRemainingText = remainingText.replace(bareUrlRegex, '');
    if (cleanedRemainingText) {
      parts.push(<span key={`text-${key}-${lastIndex}`}>{cleanedRemainingText}</span>);
    }
  }
  return parts.length > 0 ? parts : text;
};

// Function to render text with markdown formatting
export const renderMarkdownText = (text, key) => {
  const parts = parseMarkdown(text);
  return (
    <span key={key}>
      {parts.map((part, i) => 
        part.type === 'bold' ? (
          <strong key={i}>{renderTextWithLinks(part.content, `${key}-${i}`)}</strong>
        ) : (
          <span key={i}>{renderTextWithLinks(part.content, `${key}-${i}`)}</span>
        )
      )}
    </span>
  );
};

// Function to format time in seconds to readable format
export const formatResponseTime = (seconds) => {
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  if (hours > 0) {
    return `${hours}h: ${minutes}m: ${secs}s`;
  } else {
    return `${minutes}m: ${secs}s`;
  }
};

// Function to format LLM response text - show raw results as received
export const formatResponseText = (text) => {
  if (!text) return '';
  return <div style={{ whiteSpace: 'pre-wrap' }}>{text}</div>;
};